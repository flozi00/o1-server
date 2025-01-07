from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000", api_key="x")

resp = client.chat.completions.create(
    model="mock-gpt-model",
    messages=[
        {"role": "system", "content": "Hello, how can I help you today?"},
        {"role": "user", "content": "If 1+2 = 3, what is 2+2?"},
    ],
    temperature=0.1,
    stream=True,
)

for chunk in resp:
    print(chunk)
