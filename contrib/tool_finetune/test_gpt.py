from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://www.apigptopen.xyz/v1", 
    api_key="sk-1y32BUDy6ZHG5Qvf3aBb2305C04f48F4Ae5f3727C9Ab0f6a",
    http_client=httpx.Client(
        base_url="https://www.apigptopen.xyz/v1",
        follow_redirects=True,
    ),
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(completion)
print(completion.choices[0].message.content)
print(completion.usage)
print(completion.usage.total_tokens)
print('type of content:', type(completion.choices[0].message.content))
print('type of usage:', type(completion.usage))