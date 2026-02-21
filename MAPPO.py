import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions.transforms import TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.transforms as T
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## Buffer ##################################
class RolloutBuffer:
    def __init__(self):
        # 分别存储每个智能体的数据
        self.states_A = []
        self.states_B = []
        self.states_C = []
        
        self.actions_A = []
        self.actions_B = []
        self.actions_C = []
        
        self.logprobs_A = []
        self.logprobs_B = []
        self.logprobs_C = []
        
        # 共享数据
        self.state_globals = [] # 集中式 Critic 的输入
        self.rewards = []
        self.is_terminals = []
        self.state_values = [] # 集中式 Critic 的输出值
    
    def clear(self):
        del self.states_A[:]
        del self.states_B[:]
        del self.states_C[:]
        
        del self.actions_A[:]
        del self.actions_B[:]
        del self.actions_C[:]
        
        del self.logprobs_A[:]
        del self.logprobs_B[:]
        del self.logprobs_C[:]
        
        del self.state_globals[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]


################################## Networks ##################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(Actor, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        
        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            pass

    def forward(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            base_dist = MultivariateNormal(action_mean, cov_mat)
            dist = TransformedDistribution(base_dist, TanhTransform(cache_size=1))
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        return dist

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            base_dist = MultivariateNormal(action_mean, cov_mat)
            dist = TransformedDistribution(base_dist, TanhTransform(cache_size=1))
            
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = base_dist.entropy() if self.has_continuous_action_space else dist.entropy()
        
        return action_logprobs, dist_entropy


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim):
        super(CentralizedCritic, self).__init__()
        # Centralized Critic 输入维度是所有 agent state 维度的总和
        self.critic = nn.Sequential(
                        nn.LayerNorm(global_state_dim),
                        nn.Linear(global_state_dim, 256),
                        nn.Tanh(),
                        nn.Linear(256, 256),
                        nn.Tanh(),
                        nn.Linear(256, 1)
                    )
    
    def forward(self, global_state):
        return self.critic(global_state)


################################## MAPPO Class ##################################

class MAPPO:
    def __init__(self, state_dim_A, action_dim_A, 
                 state_dim_B, action_dim_B, 
                 state_dim_C, action_dim_C,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        # --- 初始化 Actors (策略网络) ---
        self.actor_A = Actor(state_dim_A, action_dim_A, has_continuous_action_space, action_std_init).to(device)
        self.actor_B = Actor(state_dim_B, action_dim_B, has_continuous_action_space, action_std_init).to(device)
        self.actor_C = Actor(state_dim_C, action_dim_C, has_continuous_action_space, action_std_init).to(device)
        
        # --- 初始化 Centralized Critic (价值网络) ---
        # Critic 输入维度 = dim(A) + dim(B) + dim(C)
        global_state_dim = state_dim_A + state_dim_B + state_dim_C
        self.critic_central = CentralizedCritic(global_state_dim).to(device)

        # --- Optimizers ---
        # 可以用一个优化器优化所有参数，也可以分开。这里为了清晰，将 Actor 和 Critic 分开
        self.optimizer_actor = torch.optim.Adam([
            {'params': self.actor_A.parameters(), 'lr': lr_actor},
            {'params': self.actor_B.parameters(), 'lr': lr_actor},
            {'params': self.actor_C.parameters(), 'lr': lr_actor}
        ])
        
        self.optimizer_critic = torch.optim.Adam(self.critic_central.parameters(), lr=lr_critic)

        # --- Old Policies for PPO Clipping ---
        self.actor_A_old = Actor(state_dim_A, action_dim_A, has_continuous_action_space, action_std_init).to(device)
        self.actor_B_old = Actor(state_dim_B, action_dim_B, has_continuous_action_space, action_std_init).to(device)
        self.actor_C_old = Actor(state_dim_C, action_dim_C, has_continuous_action_space, action_std_init).to(device)
        
        self.actor_A_old.load_state_dict(self.actor_A.state_dict())
        self.actor_B_old.load_state_dict(self.actor_B.state_dict())
        self.actor_C_old.load_state_dict(self.actor_C.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.actor_A.set_action_std(new_action_std)
            self.actor_B.set_action_std(new_action_std)
            self.actor_C.set_action_std(new_action_std)
            
            self.actor_A_old.set_action_std(new_action_std)
            self.actor_B_old.set_action_std(new_action_std)
            self.actor_C_old.set_action_std(new_action_std)
        else:
            print("WARNING : Calling MAPPO::set_action_std() on discrete action space policy")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling MAPPO::decay_action_std() on discrete action space policy")

    def select_action(self, state_A, state_B, state_C):
        """
        分别对 A, B, C 选择动作，并计算当前全局状态的 Value
        """
        # 处理状态格式
        s_A = torch.FloatTensor(state_A).to(device)
        s_B = torch.FloatTensor(state_B).to(device)
        s_C = torch.FloatTensor(state_C).to(device)
        
        # 拼接全局状态用于 Critic
        s_global = torch.cat([s_A, s_B, s_C], dim=-1)

        with torch.no_grad():
            action_A, logprob_A = self.actor_A_old.act(s_A)
            action_B, logprob_B = self.actor_B_old.act(s_B)
            action_C, logprob_C = self.actor_C_old.act(s_C)
            
            state_val = self.critic_central(s_global)

        # 存储到 Buffer
        self.buffer.states_A.append(s_A)
        self.buffer.states_B.append(s_B)
        self.buffer.states_C.append(s_C)
        self.buffer.state_globals.append(s_global) # 存全局状态
        
        self.buffer.actions_A.append(action_A)
        self.buffer.actions_B.append(action_B)
        self.buffer.actions_C.append(action_C)
        
        self.buffer.logprobs_A.append(logprob_A)
        self.buffer.logprobs_B.append(logprob_B)
        self.buffer.logprobs_C.append(logprob_C)
        
        self.buffer.state_values.append(state_val)

        # 返回 numpy 格式动作供环境使用
        return (action_A.detach().cpu().numpy().flatten(), 
                action_B.detach().cpu().numpy().flatten(), 
                action_C.detach().cpu().numpy().flatten())

    def update(self):
        # 1. 计算 Monte Carlo Returns (所有 Agent 共享 Reward)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2. 转换 Buffer 数据为 Tensor
        # States
        old_states_A = torch.squeeze(torch.stack(self.buffer.states_A, dim=0)).detach().to(device)
        old_states_B = torch.squeeze(torch.stack(self.buffer.states_B, dim=0)).detach().to(device)
        old_states_C = torch.squeeze(torch.stack(self.buffer.states_C, dim=0)).detach().to(device)
        old_state_globals = torch.squeeze(torch.stack(self.buffer.state_globals, dim=0)).detach().to(device)
        
        # Actions
        old_actions_A = torch.squeeze(torch.stack(self.buffer.actions_A, dim=0)).detach().to(device)
        old_actions_B = torch.squeeze(torch.stack(self.buffer.actions_B, dim=0)).detach().to(device)
        old_actions_C = torch.squeeze(torch.stack(self.buffer.actions_C, dim=0)).detach().to(device)
        
        # Logprobs
        old_logprobs_A = torch.squeeze(torch.stack(self.buffer.logprobs_A, dim=0)).detach().to(device)
        old_logprobs_B = torch.squeeze(torch.stack(self.buffer.logprobs_B, dim=0)).detach().to(device)
        old_logprobs_C = torch.squeeze(torch.stack(self.buffer.logprobs_C, dim=0)).detach().to(device)
        
        # State Values (from Centralized Critic)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # 3. 计算 Advantage (基于 Centralized Critic)
        # Adv = Reward - V(S_global)
        advantages = rewards.detach() - old_state_values.detach()

        # 4. Optimize Policy for K epochs
        for _ in range(self.K_epochs):

            # --- Evaluate Old Actions ---
            # 分别评估每个 Actor
            logprobs_A, dist_entropy_A = self.actor_A.evaluate(old_states_A, old_actions_A)
            logprobs_B, dist_entropy_B = self.actor_B.evaluate(old_states_B, old_actions_B)
            logprobs_C, dist_entropy_C = self.actor_C.evaluate(old_states_C, old_actions_C)
            
            # --- Evaluate Global State Value ---
            state_values = self.critic_central(old_state_globals)
            state_values = torch.squeeze(state_values)
            
            # --- Calculate Ratios ---
            ratios_A = torch.exp(logprobs_A - old_logprobs_A.detach())
            ratios_B = torch.exp(logprobs_B - old_logprobs_B.detach())
            ratios_C = torch.exp(logprobs_C - old_logprobs_C.detach())

            # --- Calculate Surrogate Loss (PPO Clip) ---
            # 所有 Actor 共享 Advantage
            def compute_actor_loss(ratios, advantages):
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                return -torch.min(surr1, surr2).mean()

            loss_actor_A = compute_actor_loss(ratios_A, advantages)
            loss_actor_B = compute_actor_loss(ratios_B, advantages)
            loss_actor_C = compute_actor_loss(ratios_C, advantages)
            
            # --- Calculate Critic Loss ---
            loss_critic = 0.5 * self.MseLoss(state_values, rewards)
            
            # --- Total Loss ---
            # Entropy 用于鼓励探索
            entropy_loss = dist_entropy_A.mean() + dist_entropy_B.mean() + dist_entropy_C.mean()
            
            total_loss_actor = loss_actor_A + loss_actor_B + loss_actor_C - 0.005 * entropy_loss
            
            # --- Update Steps ---
            self.optimizer_actor.zero_grad()
            total_loss_actor.backward()
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()
            
        # 5. Copy new weights into old policy
        self.actor_A_old.load_state_dict(self.actor_A.state_dict())
        self.actor_B_old.load_state_dict(self.actor_B.state_dict())
        self.actor_C_old.load_state_dict(self.actor_C.state_dict())

        # 6. Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save({
            'actor_A': self.actor_A_old.state_dict(),
            'actor_B': self.actor_B_old.state_dict(),
            'actor_C': self.actor_C_old.state_dict(),
            'critic_central': self.critic_central.state_dict(),
            'action_std': self.action_std,
        }, checkpoint_path)
   
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        self.actor_A.load_state_dict(checkpoint['actor_A'])
        self.actor_A_old.load_state_dict(checkpoint['actor_A'])
        
        self.actor_B.load_state_dict(checkpoint['actor_B'])
        self.actor_B_old.load_state_dict(checkpoint['actor_B'])
        
        self.actor_C.load_state_dict(checkpoint['actor_C'])
        self.actor_C_old.load_state_dict(checkpoint['actor_C'])
        
        self.critic_central.load_state_dict(checkpoint['critic_central'])
        
        if 'action_std' in checkpoint:
            self.action_std = checkpoint['action_std']
            self.set_action_std(self.action_std)
            print(f"Loaded and set action_std to: {self.action_std:.4f}")