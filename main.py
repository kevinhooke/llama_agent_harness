import requests

llama_base_url = "http://192.168.4.201:8090/v1/chat/completions"

input = input("prompt: ")

response = requests.post(
    llama_base_url,
    json={"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": input}]},
)

# print(response.json())
if response.status_code == 200:
    print(f"response: {response.json()['choices'][0]['message']['content']}")
else:
    print(f"error: {response.status_code}, response: {response.text}")
