import requests

llama_base_url = "http://192.168.4.201:8090/v1/chat/completions"

# array of messages, each interaction is appended to maintain consersation history
messages = []

while True:
    prompt = input("prompt: ")

    if prompt not in ("exit", "/exit"):
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            llama_base_url,
            json={
                "model": "qwen3.6-35b-a3b",
                "messages": messages,
            },
        )

        # print(response.json())
        if response.status_code == 200:
            response_text = response.json()["choices"][0]["message"]["content"]
            print(f"response: {response_text}")
            print()

            # keep the reponse in context
            messages.append({"role": "assistant", "content": response_text})
        else:
            print(f"error: {response.status_code}, response: {response.text}")
    else:
        break

print("Exiting...")
