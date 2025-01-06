from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000", api_key="d7f30f58a96ce692a80431e087ae635ca4049e949cb06b3da3ab596973ab221f")

resp = client.chat.completions.create(
    model="mock-gpt-model",
    messages=[
        {"role": "system", "content": "Hello, how can I help you today?"},
        {"role": "user", "content": "Fred überholt bei einem Wettrennen den 3. Platz. Wieviele Personen sind noch vor ihm?"},
    ],
    temperature=0.1,
    stream=True,
)

for chunk in resp:
    print(chunk)