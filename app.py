# app.py
import subprocess
import sys
import threading
import json
import re
import webbrowser
import os
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from topology_server import extract_topology_from_env
from Qwen import call_qwen

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')

# 获取当前工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_system_prompt():
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ 警告：未找到 {PROMPT_FILE_PATH}，使用默认提示词。")
        return "你是一个有用的助手。"
    
SYSTEM_PROMPT = load_system_prompt()

# 启动 test.py 子进程
proc = subprocess.Popen(
    [sys.executable, "test.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,          # ← 分离 stderr，不混合到 stdout
    stdin=subprocess.PIPE,           # ← 添加 stdin 管道用于决策通信
    encoding='utf-8',                # ← 显式指定编码
    errors='replace',                # ← 遇到非法字符时替换（避免崩溃）
    bufsize=1,
    cwd=script_dir                   # ← 指定工作目录
)

def monitor_simulation():
    """监听 test.py 的 stdout 和 stderr,提取 SIM_STATE 和 LLM_RESPONSE"""
    # 读取输出 stdout
    def read_stdout():
        for line in proc.stdout:
            print(f"[STDOUT] {line.rstrip()}", file=sys.stdout, flush=True)
            
            # 处理 LLM 响应
            if "LLM_RESPONSE:" in line:
                try:
                    json_str = line.split("LLM_RESPONSE:", 1)[1].strip()
                    response = json.loads(json_str)
                    socketio.emit('llm_response', response)
                except Exception as e:
                    print(f"[ERROR] LLM Parse error: {e}", flush=True)
            
            # 处理状态更新
            if "SIM_STATE:" in line:
                try:
                    json_str = line.split("SIM_STATE:", 1)[1].strip()
                    state = json.loads(json_str)
                    socketio.emit('update_state', state)    # 发送state到前端
                except Exception as e:
                    print(f"[ERROR] State parse error: {e}", flush=True)

    # 读取 stderr
    def read_stderr():
        for line in proc.stderr:
            print(f"[STDERR] {line.rstrip()}", file=sys.stderr, flush=True)
    
    # 分别启动两个线程读取 stdout 和 stderr
    t_stdout = threading.Thread(target=read_stdout, daemon=True)
    t_stderr = threading.Thread(target=read_stderr, daemon=True)
    t_stdout.start()
    t_stderr.start()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_decision')    # 装饰器，监听前端发送的 'send_decision' 事件
def receive_decision(data):
    """
    接收来自网页前端的策略决策，通过 stdin 管道发送给子进程
    data = {
        'policy': 'PARS'/'PPO'/'SAC'
    }
    """
    
    try:
        chosen_policy = data.get('policy')
        if chosen_policy not in ['PARS', 'PPO', 'SAC']:
            print(f"[ERROR] Invalid policy: {chosen_policy}", flush=True)
            socketio.emit('decision_received', {
                'status': 'error',
                'message': 'Invalid policy'
            })
            return {'status': 'error', 'message': 'Invalid policy'}
        
        # 通过 stdin 管道发送决策给子进程
        try:
            print(f"[APP] 正在发送决策到子进程: {chosen_policy}", flush=True)
            proc.stdin.write(chosen_policy + '\n')  # 写入test.py的标准输入
            proc.stdin.flush()  # 刷新输入缓冲区
            print(f"[APP] ✅ 决策已发送: {chosen_policy}", flush=True)
            
        except Exception as e:
            print(f"[ERROR] 发送决策失败: {e}", flush=True)
            socketio.emit('decision_received', {
                'status': 'error',
                'message': f'发送失败: {e}'
            })
            return {'status': 'error', 'message': str(e)}
        
        # 通知前端决策已接收
        socketio.emit('decision_received', {
            'status': 'success',
            'policy': chosen_policy,
            'message': f'已切换至 {chosen_policy} 策略'
        })
        
        return {'status': 'success', 'policy': chosen_policy}
    except Exception as e:
        print(f"[ERROR] 处理决策时出错: {e}", flush=True)
        socketio.emit('decision_received', {
            'status': 'error',
            'message': str(e)
        })
        return {'status': 'error', 'message': str(e)}
    
@socketio.on('send_chat_message')
def handle_chat_message(json_data):
    user_msg = json_data.get('message')
    context = json_data.get('context', {})
    
    # 1. 构建 Prompt (结合当前电网状态)
    current_state= f"""
    
    当前系统状态：
    - Step: {context.get('step')}
    - 策略：{context.get('policy')}
    - 低电压母线：{context.get('lowBuses')}
    - 高电压母线：{context.get('highBuses')}
    
    请基于上述实时数据，简洁地回答用户问题。如果用户问的是通用知识，直接回答即可。
    """
    
    # 2. 调用你的 LLM 函数 (复用之前的 llm_client 或 test.py 中的逻辑)
    # 假设你有一个函数 get_llm_response(prompt)
    full_prompt = SYSTEM_PROMPT + current_state + "\n用户问题：" + user_msg
    # 3. 流式推送到前端
    explanation = ""
    for chunk in call_qwen(full_prompt):
        explanation += chunk
        socketio.emit('receive_chat_response', {'text': chunk})


@app.route('/api/topology')
def api_topology():
    # 如果 topology_server 不可用则返回一个小的静态回退拓扑，方便前端调试
    if extract_topology_from_env is None:
        print('[WARN] topology extractor not available, returning fallback topology', flush=True)
        sample = {
            'nodes': [
                {'id': '800', 'name': '800', 'x': 0, 'y': 0},
                {'id': '802', 'name': '802', 'x': 40, 'y': 0},
                {'id': '806', 'name': '806', 'x': 80, 'y': 0},
                {'id': '850', 'name': '850', 'x': 120, 'y': 0}
            ],
            'links': [
                {'source': '800', 'target': '802'},
                {'source': '802', 'target': '806'},
                {'source': '806', 'target': '850'}
            ]
        }
        return jsonify(sample)

    try:
        data = extract_topology_from_env()
        return jsonify(data)
    except Exception as e:
        # 打印详细错误以便调试，并返回回退拓扑
        import traceback
        print(f"[ERROR] /api/topology failed: {e}", flush=True)
        traceback.print_exc()
        sample = {
            'nodes': [
                {'id': '800', 'name': '800', 'x': 0, 'y': 0},
                {'id': '802', 'name': '802', 'x': 40, 'y': 0},
                {'id': '806', 'name': '806', 'x': 80, 'y': 0},
                {'id': '850', 'name': '850', 'x': 120, 'y': 0}
            ],
            'links': [
                {'source': '800', 'target': '802'},
                {'source': '802', 'target': '806'},
                {'source': '806', 'target': '850'}
            ]
        }
        return jsonify(sample)

if __name__ == '__main__':
    # 启动监听线程
    threading.Thread(target=monitor_simulation, daemon=True).start()
    
    # 打开浏览器
    webbrowser.open("http://127.0.0.1:5000")
    
    # 启动 Web 服务
    socketio.run(app, host='127.0.0.1', port=5000)