from openai import OpenAI

client = OpenAI(
    api_key="sk-a628e96d5250432b8991852188c5a505",
    base_url="https://api.deepseek.com"
)

def call_LLM(prompt: str, max_tokens=800):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.8,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"[LLM Error] {str(e)}"


if __name__ == "__main__":
    test_prompt = "请简要介绍一下transformer的基本原理。"
    print("开始流式输出：")
    full_response = ""

    for chunk in call_LLM(test_prompt):
        print(chunk, end="", flush=True)
        full_response += chunk

    print("\n\n--- 完整回复接收完毕 ---")
    print(f"完整回复: {full_response}")
