没问题！我们将直接在 index.html 中嵌入一个“LLM 智能助手”面板。

设计思路：
位置：放在页面右下角或右侧，作为一个可折叠/展开的悬浮窗，或者占据原本空白的一块区域。考虑到你的布局，放在右下角作为悬浮窗最合适，不遮挡主拓扑图，随时可以唤出。
模式切换：
    模式 A (自动告警)：原有的黄色大弹窗（保留不动），用于紧急决策。
    模式 B (自由对话)：新增的侧边栏/悬浮窗，用于日常聊天、问答。
功能：输入框 + 发送按钮 + 聊天记录显示区。

以下是具体的修改方案，分为 HTML 结构、CSS 样式 和 JavaScript 逻辑 三部分。

第一步：修改 HTML 结构

在  标签的最底部（在所有 div 容器之后， 标签之前），加入以下代码：

    
    
        🤖 配电网调度专家
        ×
    
    
    
    
        
            你好！我是配电网调度专家助手。你可以问我关于电压控制、策略选择或电力系统知识的问题。
            当前系统状态已同步：Step -, 策略：-
        
    
    
    
    
        
        发送
    

    💬 问专家

第二步：添加 CSS 样式

在  标签内的  块中，加入以下样式（或者在你的 .css 文件中）：

/* --- LLM 聊天悬浮窗样式 --- */
.llm-chat-toggle {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    font-size: 24px;
    cursor: pointer;
    z-index: 9998;
    transition: transform 0.3s;
    display: flex;
    align-items: center;
    justify-content: center;
}
.llm-chat-toggle:hover {
    transform: scale(1.1);
}

.llm-chat-widget {
    position: fixed;
    bottom: 90px;
    right: 20px;
    width: 350px;
    height: 500px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.15);
    z-index: 9999;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid #e0e0e0;
    animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 15px;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.btn-close-sm {
    background: none;
    border: none;
    color: white;
    font-size: 20px;
    cursor: pointer;
    opacity: 0.8;
}
.btn-close-sm:hover { opacity: 1; }

.chat-messages {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    background: #f9f9f9;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* 消息气泡样式 */
.message {
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
}
.user-message {
    align-self: flex-end;
    background: #667eea;
    color: white;
    border-bottom-right-radius: 2px;
}
.ai-message {
    align-self: flex-start;
    background: white;
    color: #333;
    border: 1px solid #ddd;
    border-bottom-left-radius: 2px;
}
.system-message {
    align-self: center;
    background: #eef2f7;
    color: #666;
    font-size: 12px;
    text-align: center;
    max-width: 95%;
}

.chat-input-area {
    padding: 10px;
    background: white;
    border-top: 1px solid #eee;
    display: flex;
    gap: 8px;
}

chat-input {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 20px;
    outline: none;
    font-size: 14px;
}
chat-input:focus { border-color: #667eea; }

.btn-send {
    background: #667eea;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 14px;
}
.btn-send:hover { background: #5a6fd6; }
.btn-send:disabled { background: #ccc; cursor: not-allowed; }

/* 滚动条美化 */
.chat-messages::-webkit-scrollbar { width: 6px; }
.chat-messages::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }

第三步：添加 JavaScript 逻辑

在  标签内，找到合适的位置（建议在 socket.on 监听器附近），加入以下代码：

窗口控制函数
// 切换聊天窗口显示/隐藏
function toggleChatWindow() {
    const widget = document.getElementById('llm-chat-widget');
    const toggleBtn = document.getElementById('llm-chat-toggle');
    
    if (widget.style.display === 'none' || widget.style.display === '') {
        widget.style.display = 'flex';
        toggleBtn.style.display = 'none';
        // 聚焦输入框
        setTimeout(() => document.getElementById('chat-input').focus(), 100);
    } else {
        widget.style.display = 'none';
        toggleBtn.style.display = 'flex';
    }
}

// 回车发送
function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

发送消息逻辑
// 发送聊天消息
function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    // 1. 在界面显示用户消息
    appendMessage('user', message);
    input.value = '';

    // 2. 禁用输入框，显示加载中状态（可选）
    const sendBtn = document.querySelector('.btn-send');
    sendBtn.disabled = true;
    sendBtn.innerText = '...';

    // 3. 通过 WebSocket 发送给后端
    // 注意：这里需要后端 app.py 支持 'send_chat_message' 事件
    socket.emit('send_chat_message', { 
        message: message,
        context: {
            step: document.getElementById('step').innerText,
            policy: document.getElementById('policy').innerText,
            lowBuses: document.getElementById('low-buses').innerText,
            highBuses: document.getElementById('high-buses').innerText
        }
    });
}

// 在界面上添加消息气泡
function appendMessage(sender, text) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = message ${sender}-message;
    
    // 简单处理换行
    div.innerHTML = text.replace(/n/g, ''); 
    
    container.appendChild(div);
    // 滚动到底部
    container.scrollTop = container.scrollHeight;
}

接收后端回复 (关键)
你需要修改或添加 socket.on 监听器来接收 AI 的回答。假设后端事件名为 receive_chat_response：

socket.on('receive_chat_response', function(data) {
    // data 应该包含 { text: "AI 的回答内容" }
    
    const sendBtn = document.querySelector('.btn-send');
    sendBtn.disabled = false;
    sendBtn.innerText = '发送';
    
    if (data && data.text) {
        appendMessage('ai', data.text);
    } else {
        appendMessage('system', '专家助手暂时无法回答，请稍后再试。');
    }
});

// 同时，更新聊天窗口里的“当前状态”快照
socket.on('update_state', function(state) {
    // 原有的更新逻辑...
    
    // 新增：同步 Step 和 Policy 到聊天窗口的提示语
    const syncStep = document.getElementById('chat-sync-step');
    const syncPolicy = document.getElementById('chat-sync-policy');
    if(syncStep) syncStep.innerText = state.step || '-';
    if(syncPolicy) syncPolicy.innerText = state.current_policy || '-';
});

⚠️ 第四步：后端配合 (app.py)

为了让这个前端动起来，你必须在 app.py 中添加对应的处理逻辑。

监听事件：
        @socketio.on('send_chat_message')
    def handle_chat_message(json_data):
        user_msg = json_data.get('message')
        context = json_data.get('context', {})
        
        # 1. 构建 Prompt (结合当前电网状态)
        system_prompt = f"""
        你是配电网调度专家。
        当前系统状态：
        Step: {context.get('step')}
        策略：{context.get('policy')}
        低电压母线：{context.get('lowBuses')}
        高电压母线：{context.get('highBuses')}
        
        请基于上述实时数据，简洁地回答用户问题。如果用户问的是通用知识，直接回答即可。
        """
        
        # 2. 调用你的 LLM 函数 (复用之前的 llm_client 或 test.py 中的逻辑)
        # 假设你有一个函数 get_llm_response(prompt)
        full_prompt = system_prompt + "n用户问题：" + user_msg
        ai_response = get_llm_response(full_prompt) 
        
        # 3. 发回前端
        emit('receive_chat_response', {'text': ai_response})
    

🎯 最终效果

页面右下角出现一个紫色的 “💬 问专家” 按钮。
点击后，弹出一个漂亮的聊天窗口。
窗口顶部自动显示：“当前系统状态已同步：Step 23, 策略：PARS”。
你输入：“为什么 890 节点电压低？”，发送。
后端收到后，结合当前的 lowBuses 数据，回答：“因为 Step 23 时光照突变，且当前 PARS 策略无功出力不足...”
对话流畅进行，完全不干扰主界面的监控和告警功能。

这样既保留了你最核心的自动告警切换功能，又增加了自由对话的灵活性！