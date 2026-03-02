import dashscope
from dashscope import Generation

# 设置你的 API Key（建议用环境变量）
dashscope.api_key = "sk-2690c2c622834a6ebb6ec9f80a9ee9c8"

def call_qwen(prompt: str, max_tokens=800):
    try:
        responses = Generation.call(
            model="qwen-max",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.8,
            stream=True,
            result_format='message'
        )
        
        last_content = ""  # 记录上一次的内容
        for response in responses:
            if response.status_code == 200:
                if hasattr(response.output, 'choices') and response.output.choices:
                    # 获取当前chunk的完整内容
                    current_content = response.output.choices[0].message.content or ""
                    
                    # 只返回新增的部分
                    if len(current_content) > len(last_content):
                        new_content = current_content[len(last_content):]
                        yield new_content
                        last_content = current_content
                elif hasattr(response.output, 'text'):
                    current_text = response.output.text or ""
                    if len(current_text) > len(last_content):
                        new_text = current_text[len(last_content):]
                        yield new_text
                        last_content = current_text
            
            else:
                yield f'[LLM Error] API returned status code {response.status_code}: {response.output}'
                break
 
    except Exception as e:
        yield f"[LLM Error] {str(e)}"


if __name__ == "__main__":
    test_prompt = "请简要介绍一下transformer的基本原理。"
    print("开始流式输出：")
    full_response = ""
    
    for chunk in call_qwen(test_prompt):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n\n--- 完整回复接收完毕 ---")
    print(f"完整回复: {full_response}")