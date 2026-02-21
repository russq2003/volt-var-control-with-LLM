
import dashscope
from dashscope import Generation

# 设置你的 API Key（建议用环境变量）
dashscope.api_key = "sk-2690c2c622834a6ebb6ec9f80a9ee9c8"

def call_qwen(prompt: str, max_tokens=800) -> str:
    try:
        response = Generation.call(
            model="qwen-max",          # 或 qwen-plus, qwen-turbo（更便宜更快）
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,           # 降低随机性，适合解释任务
            top_p=0.8
        )
        return response.output.text.strip()
    except Exception as e:
        return f"[LLM Error] {str(e)}"


if __name__ == "__main__":
    test_prompt = "请简要介绍一下transformer的基本原理。"
    print(call_qwen(test_prompt, max_tokens=300))