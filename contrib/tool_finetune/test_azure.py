import os
import requests
import base64

# Configuration
API_KEY = os.environ.get("AZUREAI_API_KEY")
ENDPOINT = os.environ.get("AZUREAI_ENDPOINT_URL")

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Payload for the request
payload = {
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "你是一个帮助用户查找信息的 AI 助手。"
        }
      ]
    },
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "我想知道墨尔本的天气。"
            }
        ]
    }
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 800
}

ENDPOINT = "https://u3641-m39xdy74-australiaeast.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview"

# Send request
try:
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
except requests.RequestException as e:
    raise SystemExit(f"Failed to make the request. Error: {e}")


# Handle the response as needed (e.g., print or process)
print(response.json())
usage = response.json()['usage']
prompt_tokens = usage['prompt_tokens']
completion_tokens = usage['completion_tokens']
total_tokens_used = usage['total_tokens']
print("Type of response:", type(response)) 
print("Type of response json:", type(response.json()))  
message = response.json()['choices'][0]['message']['content'].strip()
print("Prompt tokens:", prompt_tokens)
print("Completion tokens:", completion_tokens)
print("Total tokens used:", total_tokens_used)

print("Response:", message)