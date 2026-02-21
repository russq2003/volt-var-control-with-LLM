# topology_server.py
import os
import sys
from flask import Flask, render_template, jsonify

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_register import make_env

app = Flask(__name__, template_folder='templates')

# ===== 配置你的环境 =====
ENV_NAME = '34Bus'  # ←←← 请根据你的实际环境修改！
# ======================

topology_data = None

def extract_topology_from_env():
    """从你的 env 中提取拓扑"""
    print(f"[INFO] Loading environment: {ENV_NAME}")
    
    # 构造一个完整的 args 对象（模拟训练时的参数）
    class Args:
        env_name = ENV_NAME
        seed = 509
        policy_type = 'LSTM'
        policy_network_size = [32, 32]
        reward_weight = [0.1, 200, 0, 10, 0, 0]
        buses_to_plot = []
        loads_to_plot = []
        plot_phase = 1
        PVmode = 1
        gen_ls_csv = False  # ←←← 关键！设为 False 避免生成 CSV
        stack_num = 1
        PPO_model_dir = r"D:/powergym_standard_version/powergym_standard_version/MAPPO_model"
        SAC_model_dir = r"D:/powergym_standard_version/powergym_standard_version/MASAC_model"
        system_name = ENV_NAME  # 有些地方可能用这个

    args = Args()
    
    # 创建环境（只初始化，不仿真）
    env = make_env(args, ENV_NAME, worker_idx=0)
    
    nodes = []
    links = []
    all_buses = set()

    manual_coords = {
    # 主干线（从左到右）
    
    "sourcebus":[-50,0],
    "800": [0, 0],
    "802": [40, 0],
    "806": [80, 0],
    "808": [120, 0],
    "810": [120, -40], 
    "812": [160, 0],
    "814": [200, 0],   
    "814r": [240, 0],  
    "850": [280, 0],
    "816": [320, 0],
    "818": [320, -30], 
    "820": [320, -60],
    "822": [320, -90],
    
    "824": [380, 0],   
    "826": [420, 0],
    "828": [380, 100], 
    "830": [430, 100],
    "854": [480, 100], 
    "856": [530, 100],
    "832": [480, 40],  
    "852": [480, 80],  
    "852r": [480, 60],
    "858": [480, 0], 
    "864": [480, -60], 
    "888": [530, 40],  
    "890": [580, 40],

    "834": [600, 0],   
    "842": [600, -30], 
    "844": [600, -60],
    "846": [600, -90], 
    "848": [600, -120],
    "860": [650, 0],
    "836": [700, 0],
    "862": [700, -50], 
    "838": [700, -90],
    "840": [750, 0]
}

    for bus in env.all_bus_names:
        bus_lower = bus.lower()
        all_buses.add(bus_lower)
        node = {
            "id": bus_lower,
            "name": bus,
            "category": "bus"
        }
        # 使用手动坐标
        x, y = manual_coords.get(bus, (1000, len(nodes) * 50))
        node["x"] = x
        node["y"] = y
            
        nodes.append(node)
    
    # 添加线路
    for b1, b2 in env.lines.values():
        b1, b2 = b1.lower(), b2.lower()
        if b1 in all_buses and b2 in all_buses:
            links.append({"source": b1, "target": b2})
    
    # 添加变压器
    for b1, b2 in env.transformers.values():
        b1, b2 = b1.lower(), b2.lower()
        if b1 in all_buses and b2 in all_buses:
            links.append({"source": b1, "target": b2})
    
    print(f"[INFO] Extracted {len(nodes)} nodes, {len(links)} edges")
    return {"nodes": nodes, "links": links}

@app.route('/')
def index():
    return render_template('topology.html')

@app.route('/api/topology')
def get_topology():
    global topology_data
    if topology_data is None:
        topology_data = extract_topology_from_env()
    return jsonify(topology_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)