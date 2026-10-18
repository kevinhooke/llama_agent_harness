import json
import subprocess

import requests
from requests import Response

llama_base_url = "http://192.168.4.201:8090/v1/chat/completions"


def main():
    # array of messages, each interaction is appended to maintain consersation history
    messages = []

    while True:
        prompt = input("prompt: ")

        if prompt not in ("exit", "/exit"):
            messages.append({"role": "user", "content": prompt})

            response = call_model(llama_base_url, messages)

            print(response.json())
            if response.status_code == 200:
                response_text = response.json()["choices"][0]["message"]["content"]
                tool_calls = response.json()["choices"][0]["message"].get("tool_calls")
                print(f"response: {response_text}")
                if tool_calls:
                    messages = invoke_tools(tool_calls, messages)
                print()

                # keep the response in context
                messages.append({"role": "assistant", "content": response_text})
            else:
                print(f"error: {response.status_code}, response: {response.text}")
        else:
            break

    print("Exiting...")


def call_model(url: str, messages) -> Response:
    return requests.post(
        url,
        json={
            "model": "qwen3.6-35b-a3b",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a bash command on the user's machine",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The bash command to execute",
                                }
                            },
                            "required": ["command"],
                        },
                    },
                }
            ],
        },
    )


def invoke_tools(tool_calls, messages: list) -> list:
    print(f"tool_calls: {tool_calls}")
    # todo: iterate the tool calls
    tool = tool_calls[0]["function"]["name"]

    if tool == "bash":
        arguments = tool_calls[0]["function"]["arguments"]

        print(f"tool: {tool}, arguments: {arguments}")
        print("\n⚠️ WARNING ⚠️ Review this carefully!")
        print(f"Model is asking to execute: {arguments}")
        tool_permission = input("Continue, y/n?")

        if tool_permission == "y":
            command = json.loads(arguments)["command"]
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,  # capture stdout and stderr
                text=True,
                timeout=60,
            )

            print(result.returncode)  # exit code, 0 = success
            print(result.stdout)
            print(result.stderr)

            # add result to messages and send back to model
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0]["id"],
                    "content": json.dumps(
                        {
                            "exit_code": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        }
                    ),
                }
            )
            print("Sending tool output to model...")
            response = call_model(llama_base_url, messages)
            response_text = response.json()["choices"][0]["message"]["content"]
            tool_calls = response.json()["choices"][0]["message"].get("tool_calls")
            print(f"response: {response_text}")
        else:
            print(f"unknown or unexpected tool requested: {tool}")

        return messages
