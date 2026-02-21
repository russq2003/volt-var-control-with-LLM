#!/usr/bin/env python3
"""
测试 app.py 和 test.py 之间的文件通信
用于诊断决策文件路径问题
"""
import os
import sys
import subprocess
import time
import threading

# 模拟 app.py 的环境
script_dir = os.path.dirname(os.path.abspath(__file__))
DECISION_FILE = os.path.join(script_dir, '.decision.txt')

print(f"[TEST] Script dir: {script_dir}")
print(f"[TEST] Decision file path: {DECISION_FILE}")
print()

# 创建一个简单的测试脚本来模拟 test.py
test_script_content = '''
import os
import sys
import time

# 获取环境变量中的决策文件路径
decision_file = os.environ.get('DECISION_FILE')
if not decision_file:
    decision_file = os.path.join(os.getcwd(), '.decision.txt')
    print(f"[TEST_CHILD] 环境变量不存在，使用默认路径", flush=True)
else:
    print(f"[TEST_CHILD] 从环境变量获取路径: {decision_file}", flush=True)

print(f"[TEST_CHILD] 当前工作目录: {os.getcwd()}", flush=True)
print(f"[TEST_CHILD] 决策文件路径: {decision_file}", flush=True)
print(f"[TEST_CHILD] 文件是否存在: {os.path.exists(decision_file)}", flush=True)

# 等待 10 秒，检查文件是否出现
for i in range(100):
    if os.path.exists(decision_file):
        print(f"[TEST_CHILD] ✅ 第 {i*0.1:.1f} 秒时检测到文件!", flush=True)
        with open(decision_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        print(f"[TEST_CHILD] 文件内容: {content}", flush=True)
        break
    if i % 10 == 0:
        print(f"[TEST_CHILD] 等待中... ({i*0.1:.1f} 秒)", flush=True)
    time.sleep(0.1)
else:
    print(f"[TEST_CHILD] ❌ 10 秒后仍未检测到文件", flush=True)
'''

# 创建临时测试脚本
test_script_path = os.path.join(script_dir, 'TEST_CHILD.py')
with open(test_script_path, 'w', encoding='utf-8') as f:
    f.write(test_script_content)

print("[TEST] 启动子进程...")
# 启动子进程，使用与 app.py 相同的环境变量传递方式
proc = subprocess.Popen(
    [sys.executable, "TEST_CHILD.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    encoding='utf-8',
    errors='replace',
    bufsize=1,
    cwd=script_dir,
    env={**os.environ, 'DECISION_FILE': DECISION_FILE}
)

def monitor_output():
    for line in proc.stdout:
        print(line.rstrip())
    for line in proc.stderr:
        print(f"[STDERR] {line.rstrip()}")

monitor_thread = threading.Thread(target=monitor_output, daemon=True)
monitor_thread.start()

# 等待 2 秒后在父进程中写入文件
print("\n[TEST] 等待 2 秒...")
time.sleep(2)

print(f"[TEST] 正在写入决策文件: {DECISION_FILE}")
try:
    with open(DECISION_FILE, 'w', encoding='utf-8') as f:
        f.write('PPO')
        f.flush()
        try:
            os.fsync(f.fileno())
        except:
            pass
    
    print(f"[TEST] ✅ 文件已写入")
    
    # 验证文件内容
    time.sleep(0.2)
    if os.path.exists(DECISION_FILE):
        with open(DECISION_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        print(f"[TEST] ✅ 文件验证成功: {content}")
    else:
        print(f"[TEST] ❌ 文件不存在")
except Exception as e:
    print(f"[TEST] ❌ 写入失败: {e}")

# 等待子进程完成
proc.wait(timeout=15)

# 清理
try:
    os.remove(test_script_path)
    os.remove(DECISION_FILE)
except:
    pass

print("\n[TEST] 测试完成")
