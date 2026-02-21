'''负责运行配电网环境PowerGym。
执行三种控制策略 PARS (你的LSTM策略), PPO, SAC。
关键逻辑: 当检测到电压异常或电池SOC极端时,调用 LLM (Qwen) 进行分析，并暂停等待用户指令来决定是否切换策略。
通信方式: 通过标准输出 (stdout) 发送状态数据，通过标准输入 (stdin) 接收用户决策。'''

import matplotlib.pyplot as plt
import numpy as np
import random
import sys, os, time, argparse, math
import queue
import threading
from statistics import mean
from MAPPO import MAPPO
from MASAC import MASACAgent
import torch
import traceback
import json
from env_register import make_env, remove_parallel_dss
from obserfilter import RunningStat
from policy_LSTM import LinearPolicy, FullyConnectedNeuralNetworkPolicy, LSTMPolicy
from Qwen import call_qwen

sys.path.append(os.getcwd())

DAY = 2
DAY_CASES = [DAY]

PURTERBATIONS = []
for case in range(365):
    PURTERBATIONS.append([])
    for t in range(289):
        PURTERBATIONS[case].append([])
        for i in range(48):
            k = i
            np.random.seed(112)
            PURTERBATIONS[case][t].append(1 - (k / 200) * random.uniform(-1.0, 1.0))
PURTERBATIONS = np.array(PURTERBATIONS, dtype=np.float64)


def get_dims(args, worker_idx=1):
    env = make_env(args, args.env_name, worker_idx=worker_idx)
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    obs_dim = env.observation_space.shape[0]
    CRB_num = (env.cap_num, env.reg_num, env.bat_num, env.PV_num)
    print('NumCap, NumReg, NumBat, NumPV: {}'.format(CRB_num))
    print('ObsDim, ActDim: {}, {}'.format(obs_dim, sum(CRB_num)))
    
    obs_dict = env.reset(load_profile_idx=0)
    obs_dim_A = len(obs_dict['A'])
    obs_dim_B = len(obs_dict['B'])
    obs_dim_C = len(obs_dict['C'])
    
    return obs_dim, CRB_num, obs_dim_A, obs_dim_B, obs_dim_C

def trigger_llm(env):
    """
    基于环境真实状态判断是否触发 LLM。
    """
    # 检查所有母线电压（跳过无效值 0）
    for volt_list in env.obs['bus_voltages'].values():
        for v in volt_list:
            if v > 0.001 and (v < 0.95 or v > 1.05):
                return True

    # 检查电池 SOC 是否极端
    for status in env.obs['bat_statuses'].values():
        soc = status[0]
        if soc <= 0.15 or soc >= 0.85:
            return True

    return False

def llm_prompt(env, pars_action, ppo_action, sac_action, current_main_policy):

    # === 1. 提取关键状态 ===
    bus_voltages = env.obs['bus_voltages']
    low_v_buses = []
    high_v_buses = []
    for bus, v_list in bus_voltages.items():
        for v in v_list:
            if v > 0.001:  # 有效电压
                if v < 0.95:
                    low_v_buses.append(f"{bus}: {v:.3f}")
                elif v > 1.05:
                    high_v_buses.append(f"{bus}: {v:.3f}")

    bat_statuses = env.obs['bat_statuses']
    low_soc_bats = [name for name, (soc, _) in bat_statuses.items() if soc < 0.1]
    high_soc_bats = [name for name, (soc, _) in bat_statuses.items() if soc > 0.9]

    # === 2. 转换动作为自然语言描述 ===
    pars_desc = action_to_device_desc(pars_action)
    ppo_desc = action_to_device_desc(ppo_action)
    sac_desc = action_to_device_desc(sac_action)

    # === 3. 构建完整 prompt ===
    prompt = f"""
    你是一名资深配电网调度专家。当前系统出现异常，请基于以下信息提供专业建议。
    【当前电网状态】
    - 低电压母线（<0.95 p.u.): {', '.join(low_v_buses) if low_v_buses else '无'}
    - 高电压母线（>1.05 p.u.): {', '.join(high_v_buses) if high_v_buses else '无'}
    - 电池 SOC 过低（<0.15): {', '.join(low_soc_bats) if low_soc_bats else '无'}
    - 电池 SOC 过高（>0.85): {', '.join(high_soc_bats) if high_soc_bats else '无'}

    【各策略在当前时间步的控制意图】
    - 【PARS】: {pars_desc}
    - 【PPO】: {ppo_desc}
    - 【SAC】: {sac_desc}

    【当前主控策略】{current_main_policy}

    【请按以下格式回答】
    1. 诊断当前问题根本原因。
    2. 说明当前采用的是什么策略，评估当前策略是否有效。
    3. 是否建议切换策略？如是，推荐哪一个？为什么？
    4. 预期切换后的效果与风险。

    要求：
    - 必须明确指出具体设备及其动作值(例:“s850pv 发出容性无功 0.250 MVar”);
    - 分析该动作是否作用于低/高电压母线(如890号);
    - 不要使用模糊表述（如“部分调节”、“一些设备”）；
    - 不要编造未列出的设备动作；
    - 建议必须明确（不要说“视情况而定”或“可能有效”）。
    - 不一定每次给出建议的时候都要切换策略，若当前策略合理可继续执行，一定要视具体策略给出的动作值做决定。
    """
    return prompt

def action_to_device_desc(action_10d):
    """ 
    将 10 维动作向量(5 个 battery + 5 个 pv)转换为含具体数值的自然语言描述。
    输入: action_10d = [bat1, bat2, bat3, bat4, bat5, pv1, pv2, pv3, pv4, pv5]
    """
    if len(action_10d) != 10:
        raise ValueError(f"Expected 10-dimensional action, got {len(action_10d)}")

    bat_names = ['Battery.batt1', 'Battery.batt2', 'Battery.batt3', 'Battery.batt4', 'Battery.batt5']
    pv_names  = ['s846pv', 's850pv', 's858pv', 's862pv', 's854pv']
    
    desc_lines = []
    
    # 前5维：储能
    for i, name in enumerate(bat_names):
        val = action_10d[i]  # ← 现在是 [0] 到 [4]
        if abs(val) > 0.01:
            if val > 0:
                desc_lines.append(f"{name} 放电 {val:.3f} MW")
            else:
                desc_lines.append(f"{name} 充电 {abs(val):.3f} MW")
    
    # 后5维：光伏无功
    for i, name in enumerate(pv_names):
        val = action_10d[5 + i]  # ← [5] 到 [9]
        if abs(val) > 0.01:
            if val > 0:
                desc_lines.append(f"{name} 发出容性无功 {val:.3f} MVar")
            else:
                desc_lines.append(f"{name} 吸收感性无功 {abs(val):.3f} MVar")
    
    if not desc_lines:
        return "无显著设备调节动作"
    
    return ";".join(desc_lines)

def print_detailed_actions(PARS_action, PPO_action, SAC_action):
    """
    打印三种策略在储能和光伏上的具体动作值，并标注设备名。
    """
    # 设备名称
    bat_names = ['Battery.batt1', 'Battery.batt2', 'Battery.batt3', 'Battery.batt4', 'Battery.batt5']
    pv_names  = ['s846pv', 's850pv', 's858pv', 's862pv', 's854pv']
    
    # 提取动作部分（索引 8～12: battery, 13～17: pv）
    pars_bat = PARS_action[8:13]
    pars_pv  = PARS_action[13:18]
    ppo_bat  = PPO_action[8:13]
    ppo_pv   = PPO_action[13:18]
    sac_bat  = SAC_action[8:13]
    sac_pv   = SAC_action[13:18]

    print("\n【各策略具体动作值】")
    print("格式: Battery.battX (充/放电 MW) | sYYYpv (无功 MVar)")
    print("-" * 70)
    
    # 表头
    print(f"{'设备':<15} {'PARS':<12} {'PPO':<12} {'SAC':<12}")
    print("-" * 70)
    
    # 储能行
    for i, name in enumerate(bat_names):
        p_val = f"{pars_bat[i]:+.3f}"
        o_val = f"{ppo_bat[i]:+.3f}"
        s_val = f"{sac_bat[i]:+.3f}"
        print(f"{name:<15} {p_val:<12} {o_val:<12} {s_val:<12}")
    
    print("-" * 70)
    
    # 光伏行
    for i, name in enumerate(pv_names):
        p_val = f"{pars_pv[i]:+.3f}"
        o_val = f"{ppo_pv[i]:+.3f}"
        s_val = f"{sac_pv[i]:+.3f}"
        print(f"{name:<15} {p_val:<12} {o_val:<12} {s_val:<12}")
    
    print("-" * 70)
    print("注: 正值 = 放电(电池)/容性无功(光伏), 负值 = 充电/感性无功\n")

class SingleRolloutSlaver(object):

    def __init__(self, policy_params, policy_params_A, policy_params_B, policy_params_C, slaver_idx, PVmode):
        self.PVmode = PVmode
        self.env = make_env(args, args.env_name, worker_idx=slaver_idx)
        self.all_bus_names = self.env.all_bus_names
        self.policy_type = policy_params['type']

        if policy_params['type'] == 'LSTM':
            self.policy_A = LSTMPolicy(policy_params_A)
            self.policy_B = LSTMPolicy(policy_params_B)
            self.policy_C = LSTMPolicy(policy_params_C)
        else:
            raise NotImplementedError("Test script currently optimized for LSTM policies.")
        
        time.sleep(1)

    def single_rollout(self, day_case_id, 
                       new_weights_A, ob_mean_A, ob_std_A, 
                       new_weights_B, ob_mean_B, ob_std_B,
                       new_weights_C, ob_mean_C, ob_std_C, 
                       evaluate=False):

        purterbations = PURTERBATIONS[day_case_id]
        day_tuple = DAY_CASES[day_case_id]
        total_reward = 0.
        episode_steps = 0

        # initialize PPO
        lr_actor = 0.00005
        lr_critic = 0.00005
        gamma = 0.9 
        K_epochs = 15
        eps_clip = 0.15
        has_continuous = True

        mappo_agent = MAPPO(OB_DIM_A, 2,
                        OB_DIM_B, 4,
                        OB_DIM_C, 4,
                        lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous)
        
        model_path = os.path.join(args.PPO_model_dir, 'model_MAPPO.path')
        mappo_agent.load(model_path)

        # SAC
        SACagent_A = MASACAgent(OB_DIM_A, 2, OB, 10, (0, OB_DIM_A), (0, 2))
        SACagent_B = MASACAgent(OB_DIM_B, 4, OB, 10, (OB_DIM_A, OB_DIM_A+OB_DIM_B), (2, 6))
        SACagent_C = MASACAgent(OB_DIM_C, 4, OB, 10, (OB_DIM_A+OB_DIM_B, OB_DIM_A+OB_DIM_B+OB_DIM_C), (6, 10))

        SACagent_A.actor.load_state_dict(torch.load(os.path.join(args.SAC_model_dir, 'actor_A_best.pth'))['actor'])
        SACagent_B.actor.load_state_dict(torch.load(os.path.join(args.SAC_model_dir, 'actor_B_best.pth'))['actor'])
        SACagent_C.actor.load_state_dict(torch.load(os.path.join(args.SAC_model_dir, 'actor_C_best.pth'))['actor'])

        for agent in [SACagent_A, SACagent_B, SACagent_C]: agent.actor.eval()

        # PARS
        self.policy_A.update_weights(new_weights_A)
        self.policy_B.update_weights(new_weights_B)
        self.policy_C.update_weights(new_weights_C)

        self.RS_A = RunningStat(shape=(OB_DIM_A,))
        self.RS_B = RunningStat(shape=(OB_DIM_B,))
        self.RS_C = RunningStat(shape=(OB_DIM_C,))

        ob_dict = self.env.reset(load_profile_idx=day_tuple, purterbations=purterbations)
        ob_A = ob_dict['A']
        ob_B = ob_dict['B']
        ob_C = ob_dict['C']

        if self.policy_type == 'LSTM':
            self.policy_A.reset()
            self.policy_B.reset()
            self.policy_C.reset()

        if evaluate:
            vol_saves = []
            act_save = []
            bat_n = len(self.env.obs['bat_statuses'])
            soc_saves = [[] for _ in range(bat_n)]
            for bus_id in self.all_bus_names:
                vol_saves.append([[bus_id]]) # Header

        done = False
        
        current_main_policy = "PARS"
        while not done:

            time.sleep(0.5)
            
            ob_A = np.asarray(ob_A, dtype=np.float64)
            ob_B = np.asarray(ob_B, dtype=np.float64)
            ob_C = np.asarray(ob_C, dtype=np.float64)

            normal_ob_A = (ob_A - ob_mean_A) / (ob_std_A + 1e-8)
            normal_ob_B = (ob_B - ob_mean_B) / (ob_std_B + 1e-8)
            normal_ob_C = (ob_C - ob_mean_C) / (ob_std_C + 1e-8)


            action_org_pv_A, _ = self.policy_A.act(normal_ob_A)
            action_org_pv_B, _ = self.policy_B.act(normal_ob_B)
            action_org_pv_C, _ = self.policy_C.act(normal_ob_C)

            action_combined = self.combine_actions(action_org_pv_A,
                                                   action_org_pv_B,
                                                   action_org_pv_C)
            
            # PPO action
            a1, a2, a3 = mappo_agent.select_action(ob_dict['A'], ob_dict['B'], ob_dict['C'])
            PPO_action = np.concatenate(([0, 0], [16] * 6, self.combine_actions(a1, a2, a3)))

            # SAC action
            s_a1 = SACagent_A.select_action(ob_dict['A'], evaluate=True)
            s_a2 = SACagent_B.select_action(ob_dict['B'], evaluate=True)
            s_a3 = SACagent_C.select_action(ob_dict['C'], evaluate=True)
            SAC_action = np.concatenate(([0, 0], [16] * 6, self.combine_actions(s_a1, s_a2, s_a3)))

            PARS_action = np.concatenate(([0, 0, 16, 16, 16, 16, 16, 16], action_combined))

            policy_actions = {
            "PARS": PARS_action,
            "PPO": PPO_action,
            "SAC": SAC_action
            }

            selected_action = policy_actions[current_main_policy]

            triggered = trigger_llm(self.env)

            if triggered:
                print(f"\n[!] 检测到异常状态，启动 LLM 协同决策 (Step {episode_steps})", flush=True)
                print_detailed_actions(PARS_action, PPO_action, SAC_action)
                prompt = llm_prompt(self.env, action_combined, PPO_action[8:], SAC_action[8:], current_main_policy)
                explanation = call_qwen(prompt)

                print("\n【LLM 专家建议】", flush=True)
                print("-" * 40, flush=True)
                print(explanation, flush=True)
                print("-" * 40, flush=True)
                
                # 发送 LLM 响应到网页前端
                llm_response = {
                    "step": episode_steps,
                    "triggered": True,
                    "current_policy": current_main_policy,
                    "current_actions": policy_actions[current_main_policy][8:].tolist(),
                    "actions": {
                        "PARS": policy_actions["PARS"][8:].tolist(),
                        "PPO": policy_actions["PPO"][8:].tolist(),
                        "SAC": policy_actions["SAC"][8:].tolist()
                    }, 
                    "llm_explanation": explanation,
                    "device_names": {
                        "batteries": ['Battery.batt1', 'Battery.batt2', 'Battery.batt3', 'Battery.batt4', 'Battery.batt5'],
                        "pvs": ['s846pv', 's850pv', 's858pv', 's862pv', 's854pv']
                    }
                }
                print("LLM_RESPONSE:", json.dumps(llm_response), flush=True)

                # ===== 等待用户决策 =====
                # ===== 使用队列 + 线程方式读取 stdin（兼容 Windows）=====
                decision_queue = queue.Queue()
                
                def read_stdin_worker():
                    """在单独线程中阻塞读取 stdin"""
                    try:
                        line = sys.stdin.readline().strip()
                        if line:
                            decision_queue.put(line)
                    except:
                        pass
                
                # 启动读取线程（daemon 线程，主程序退出时自动关闭）
                stdin_thread = threading.Thread(target=read_stdin_worker, daemon=True)
                stdin_thread.start()
                
                wait_count = 0
                max_wait = 600  # 600 * 0.1 秒 = 60 秒
                decision_made = False
                chosen_policy = None
                
                while wait_count < max_wait and not decision_made:
                    # 尝试从队列获取决策（非阻塞）
                    try:
                        chosen_policy = decision_queue.get(timeout=0.1)  # 0.1 秒超时
                        
                        # 验证策略名称是否有效
                        if chosen_policy and chosen_policy in ['PARS', 'PPO', 'SAC']:
                            old_policy = current_main_policy
                            current_main_policy = chosen_policy
                            decision_made = True
                            print(f"[LLM] ✅ 策略切换成功: {old_policy} → {current_main_policy}", flush=True)
                            break
                        else:
                            print(f"[LLM] ⚠️ 接收到无效的策略值: '{chosen_policy}'", flush=True)
                    except queue.Empty:
                        # 队列为空，继续等待
                        pass
                    except Exception as e:
                        print(f"[LLM_DEBUG] 队列读取异常: {e}", flush=True)
                    
                    wait_count += 1
                    
                    # 每5秒打一条日志，表示还在等待
                    if wait_count % 50 == 0:
                        remaining = (max_wait - wait_count) / 10
                        print(f"[LLM] ⏳ 仍在等待决策... (剩余 {remaining:.1f} 秒)", flush=True)
                
                # 超时处理
                if not decision_made:
                    if wait_count >= max_wait:
                        print("[LLM] ⏱️ 等待超时（60秒），保持当前策略继续运行", flush=True)
                    else:
                        print("[LLM] 等待过程出错，使用当前策略继续", flush=True)
                
                print(f"[LLM] ✓ 已确认策略: {current_main_policy} | 决策状态: {'已接收' if decision_made else '超时/出错'}\n", flush=True)
            
            # extract real SOCs from env.obs['bat_statuses'] if available
            bat_statuses = self.env.obs.get('bat_statuses', {})
            soc_list = [bat_statuses.get(f'Battery.batt{i+1}', [0.5])[0] for i in range(5)]

            state = {
                        "step": episode_steps,
                        "bus_voltages": self.env.obs['bus_voltages'],
                        "socs": soc_list,
                        "actions": {
                            "PARS": PARS_action[8:].tolist(),
                            "PPO": PPO_action[8:].tolist(),
                            "SAC": SAC_action[8:].tolist(),
                        },
                        "current_policy": current_main_policy,  # ✅ 现在是最新的！
                        "trigger_llm": triggered,
                        "low_voltage_buses": [
                            bus for bus, v_list in self.env.obs['bus_voltages'].items()
                            for v in v_list if v > 0.001 and v < 0.95
                        ],
                         'high_voltage_buses': [
                            bus for bus, v_list in self.env.obs['bus_voltages'].items()
                            for v in v_list if v > 0.001 and v > 1.05
                        ]
                    }
            
            # 根据当前策略选择动作
            selected_action = policy_actions[current_main_policy]
            ob_dict, reward, done, _ = self.env.step(selected_action, purterbations=purterbations)

            ob_A = ob_dict['A']
            ob_B = ob_dict['B']
            ob_C = ob_dict['C']

            print("SIM_STATE:", json.dumps(state), flush=True)

            if evaluate:
                act_save.append(action_combined)
                i = 0
                for bus_id in self.all_bus_names:
                    if bus_id in self.env.bus_voltages:
                        vol_saves[i].append(self.env.bus_voltages[bus_id][0])
                    else:
                         vol_saves[i].append(0) # Fallback
                    i += 1
                
                num = 0
                for _, state in self.env.obs['bat_statuses'].items():
                    soc_saves[num].append(state[0])
                    num += 1

            self.env.collect_vols()
            total_reward += reward
            episode_steps += 1

            if done:
                break

        print(f"LSTM Test Finished. Total Reward: {total_reward:.2f}, Steps: {episode_steps}")

        if evaluate:
            return {'reward': total_reward, 'step': episode_steps, 'vol_saves': vol_saves, 'act_save': act_save, 'soc_saves': soc_saves}

        return {'reward': total_reward, 'step': episode_steps}
    
    def combine_actions(self, action_A, action_B, action_C):

        zone_A_devices = {'battery': ['Battery.batt2'], 'pv': ['s850pv']}
        zone_B_devices = {'battery': ['Battery.batt5', 'Battery.batt3'], 'pv': ['s854pv', 's858pv']}
        zone_C_devices = {'battery': ['Battery.batt1', 'Battery.batt4'], 'pv': ['s846pv', 's862pv']}
        
        original_bat_order = ['Battery.batt1', 'Battery.batt2', 'Battery.batt3', 'Battery.batt4', 'Battery.batt5']
        original_pv_order = ['s846pv', 's850pv', 's858pv', 's862pv', 's854pv']
        
        combined_action = []
        
        # Battery mapping
        for bat_name in original_bat_order:
            if bat_name in zone_A_devices['battery']:
                idx = zone_A_devices['battery'].index(bat_name)
                combined_action.append(action_A[idx])
            elif bat_name in zone_B_devices['battery']:
                idx = zone_B_devices['battery'].index(bat_name)
                combined_action.append(action_B[idx])
            elif bat_name in zone_C_devices['battery']:
                idx = zone_C_devices['battery'].index(bat_name)
                combined_action.append(action_C[idx])

        # PV mapping
        for pv_name in original_pv_order:
            if pv_name in zone_A_devices['pv']:
                idx = zone_A_devices['pv'].index(pv_name)
                combined_action.append(action_A[len(zone_A_devices['battery']) + idx])
            elif pv_name in zone_B_devices['pv']:
                idx = zone_B_devices['pv'].index(pv_name)
                combined_action.append(action_B[len(zone_B_devices['battery']) + idx])
            elif pv_name in zone_C_devices['pv']:
                idx = zone_C_devices['pv'].index(pv_name)
                combined_action.append(action_C[len(zone_C_devices['battery']) + idx])
        
        return np.array(combined_action)

    def close_env(self):
        self.env.close_env()


def run_episodic_random_agent(args, PVmode, worker_idx=None):
    
    purterbations = PURTERBATIONS[DAY]
    env = make_env(args, args.env_name, worker_idx=worker_idx)
    env.seed(args.seed)
    
    episode_reward = 0
    episode_steps = 0
    done = False
    
    env.reset(load_profile_idx=DAY, purterbations=purterbations)

    while not done:

        if PVmode == 0 or PVmode == 2:
            # No PV Control
            action = [0, 0, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif PVmode == 1:
            # Simple Logic (placeholder)
            action = [0, 0, 16, 16, 16, 16, 16, 16, 0, 0, 1, 1, 1, 1, 1]

        target_dim = env.cap_num + env.reg_num + env.bat_num + env.PV_num
        if len(action) < target_dim:
            action = np.concatenate((action, np.zeros(target_dim - len(action))))
        elif len(action) > target_dim:
            action = action[:target_dim]

        _, reward, done, _ = env.step(action, purterbations=purterbations)
        env.collect_vols()
        
        episode_steps += 1
        episode_reward += reward

    print(f"Baseline (Mode {PVmode}) Finished. Reward: {episode_reward:.2f}")
    return env.all_1phase_to_plot()


if __name__ == '__main__':
    print("[INIT] Starting test.py", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='34Bus')
    parser.add_argument('--seed', type=int, default=509)
    parser.add_argument('--policy_type', type=str, default='LSTM')
    parser.add_argument('--policy_network_size', type=list, default=[32, 32])
    parser.add_argument('--reward_weight', type=list, default=[0.1, 200, 0, 10, 0, 0])
    parser.add_argument('--buses_to_plot', type=list, default=[], help='buses to plot')
    parser.add_argument('--loads_to_plot', type=list, default=[], help='loads to plot')
    parser.add_argument('--plot_phase', type=int, default=1)
    parser.add_argument('--PVmode', type=int, default=1)
    parser.add_argument('--gen_ls_csv', type=bool, default=True)
    parser.add_argument('--stack_num', type=int, default=1)
    parser.add_argument('--PPO_model_dir', type=str, default=r"D:/powergym_standard_version/powergym_standard_version/MAPPO_model")
    parser.add_argument('--SAC_model_dir', type=str, default=r"D:/powergym_standard_version/powergym_standard_version/MASAC_model")
    
    args = parser.parse_args()
    print("[INIT] Arguments parsed", flush=True)

    print("[INIT] Getting environment dimensions...", flush=True)
    OB, AC, OB_DIM_A, OB_DIM_B, OB_DIM_C = get_dims(args)
    print(f"[INIT] Dimensions obtained: OB={OB}, OB_DIM_A={OB_DIM_A}, OB_DIM_B={OB_DIM_B}, OB_DIM_C={OB_DIM_C}", flush=True)

    print("\n--- Generating Baselines ---", flush=True)
    args.PVmode = 0 
    print("[BASELINE] Running Mode 0 (No PV Q control)...", flush=True)
    vol_baseline_PVnoQ, all_buses = run_episodic_random_agent(args, args.PVmode, worker_idx=None)
    print("[BASELINE] Mode 0 completed", flush=True)
    
    args.PVmode = 2
    print("[BASELINE] Running Mode 2 (No PV)...", flush=True)
    vol_baseline_NoPV, _ = run_episodic_random_agent(args, args.PVmode, worker_idx=None)
    print("[BASELINE] Mode 2 completed", flush=True)

    print("\n--- Testing Trained LSTM Agent ---", flush=True)
    args.PVmode = 1 

    policy_params = {'type': args.policy_type, 'policy_network_size': args.policy_network_size, 'ob_dim': OB, 'ac_dim': 10}
    policy_params_A = {'type': args.policy_type, 'policy_network_size': args.policy_network_size, 'ob_dim': OB_DIM_A, 'ac_dim': 2}
    policy_params_B = {'type': args.policy_type, 'policy_network_size': args.policy_network_size, 'ob_dim': OB_DIM_B, 'ac_dim': 4}
    policy_params_C = {'type': args.policy_type, 'policy_network_size': args.policy_network_size, 'ob_dim': OB_DIM_C, 'ac_dim': 4}

    print("[LSTM] Initializing SingleRolloutSlaver...", flush=True)
    slaver = SingleRolloutSlaver(policy_params=policy_params,
                                 policy_params_A=policy_params_A,
                                 policy_params_B=policy_params_B,
                                 policy_params_C=policy_params_C,
                                 slaver_idx=None,
                                 PVmode=args.PVmode)
    print("[LSTM] SingleRolloutSlaver initialized", flush=True)

    model_dir = 'D:/powergym_standard_version/powergym_standard_version/' 
    iter_num = 1430

    print(f"[MODEL] Loading models from {model_dir} at iter {iter_num}...", flush=True)
    try:
        trained_para_A = np.load(f'{model_dir}LSTM_policy_A_{iter_num}.npz', allow_pickle=True)
        trained_para_B = np.load(f'{model_dir}LSTM_policy_B_{iter_num}.npz', allow_pickle=True)
        trained_para_C = np.load(f'{model_dir}LSTM_policy_C_{iter_num}.npz', allow_pickle=True)
        print(f"[MODEL] Model files loaded successfully", flush=True)

        trained_weights_A = trained_para_A['arr_0'][0]
        trained_mean_A = trained_para_A['arr_0'][1]
        trained_std_A = trained_para_A['arr_0'][2]

        trained_weights_B = trained_para_B['arr_0'][0]
        trained_mean_B = trained_para_B['arr_0'][1]
        trained_std_B = trained_para_B['arr_0'][2]

        trained_weights_C = trained_para_C['arr_0'][0]
        trained_mean_C = trained_para_C['arr_0'][1]
        trained_std_C = trained_para_C['arr_0'][2]
        print("[MODEL] Model weights extracted", flush=True)

        print("[ROLLOUT] Starting rollout evaluation...", flush=True)
        t1 = time.time()
        result = slaver.single_rollout(0, 
                                       trained_weights_A, trained_mean_A, trained_std_A, 
                                       trained_weights_B, trained_mean_B, trained_std_B,
                                       trained_weights_C, trained_mean_C, trained_std_C,
                                       evaluate=True)
        print(f'[ROLLOUT] Rollout Time: {time.time() - t1:.4f}s', flush=True)

        vol_list = result['vol_saves']
        act_list = result['act_save']
        soc_list = result['soc_saves']
        # Emit plot-ready data as JSON to stdout so the web frontend can consume it
        try:
            plot_payload = {
                'all_buses': all_buses,
                'vol_baseline_PVnoQ': {k: list(v) for k, v in vol_baseline_PVnoQ.items()},
                'vol_baseline_NoPV': {k: list(v) for k, v in vol_baseline_NoPV.items()},
                # vol_list entries follow test.py convention: first element may be header
                'vol_list': [list(x) for x in vol_list]
            }
            print('PLOT_DATA:' + json.dumps(plot_payload), flush=True)
        except Exception as e:
            print('[WARN] Failed to serialize plot data: ' + str(e), flush=True)

        print("\nPlotting results...")
        for i in range(len(all_buses)):
            bus_name = all_buses[i]
            
            plt.figure(figsize=(10, 5))
            plt.axhline(y=1.05, linestyle='-.', color='red', linewidth=2, label='Upper Limit')
            plt.axhline(y=0.95, linestyle='-.', color='red', linewidth=2, label='Lower Limit')

            plt.plot(vol_baseline_PVnoQ[bus_name], label='PV (Q=0)', linestyle='--', alpha=0.7)
            plt.plot(vol_baseline_NoPV[bus_name], label='No PV', linestyle=':', alpha=0.7)
            
            plt.plot(vol_list[i][1:], label='PV_Q Controlled', linewidth=2, color='green')
            
            plt.legend()
            plt.xlabel('Time Steps')
            plt.ylabel('Voltage (p.u.)')
            plt.title(f'Bus {bus_name} Voltage')
            plt.grid(True)
            plt.show()

            if i >= 15:
                print("Displayed first 5 buses. Modify code to see all.")
                break

        plt.figure(figsize=(10, 5))
        act_array = np.array(act_list)
        for dim in range(min(5, act_array.shape[1])):
            plt.plot(act_array[:, dim], label=f'Action Dim {dim}')
        plt.title("Control Actions (Partial)")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 5))
        for idx, s in enumerate(soc_list):
            plt.plot(s, label=f'Battery {idx+1}')
        plt.title("Battery SOC")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        rew_list = np.load(f'{model_dir}rew_output_1500.npz',allow_pickle=True)
        rew_list1 = rew_list['arr_0'].tolist()
        plt.plot(rew_list1)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Training Reward over Episodes')
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print("\n[Error] Model files not found!")
        print(f"Please check the path: {model_dir}")
        print("Make sure 'LSTM_policy_X_1500.npz' files exist.")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
        traceback.print_exc()