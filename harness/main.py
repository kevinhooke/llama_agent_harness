import json
import os
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests import Response

CONFIG_PATH = Path(__file__).parent / "config" / "models.json"
ENV_PATH = Path(__file__).parent.parent / ".env"

"""Simple harness for interacting with LLMs via the Anthropic API style messages.

Available models are configured in config/models.json. Configure and update as needed.

Run with `poetry run harness`.

At start up, prompts user to select which model to interact with.

To use Claude models via API calls, you must have a budget for API usage (not included
in Pro by default unless you purchase credits with Claude Console), and configure an
API key in an env var, "ANTHROPIC_API_KEY". The key can be set in a .env file in the
project root (see .env.example).
"""


def main():
    # load API keys from .env; variables already set in the shell take precedence
    load_dotenv(ENV_PATH)

    # local model configs
    with open(CONFIG_PATH, encoding="utf-8") as f:
        models = json.load(f)

        selected_model = select_model(models)

        # array of messages, each interaction is appended to maintain consersation
        # history
        messages = []

        while True:
            prompt = input("prompt: ")

            if prompt not in ("exit", "/exit"):
                messages.append({"role": "user", "content": prompt})

                response = call_model(selected_model, models, messages)

                print(response.json())
                if response.status_code == 200:
                    response_text = response.json()["choices"][0]["message"]["content"]
                    tool_calls = response.json()["choices"][0]["message"].get(
                        "tool_calls"
                    )
                    print(f"response: {response_text}")
                    if tool_calls:
                        messages = invoke_tools(
                            tool_calls, selected_model, models, messages
                        )
                    print()

                    # keep the response in context
                    messages.append({"role": "assistant", "content": response_text})
                else:
                    print(f"error: {response.status_code}, response: {response.text}")
            else:
                break

        print("Exiting...")


def call_model(selected_model, models, messages) -> Response:

    model_config = models[selected_model]

    # TODO track messages state outside later
    # copy the configured body so the loaded config isn't modified
    body = dict(model_config["message_body"], messages=messages)

    # an empty tools list can be rejected by the API, so only send it when populated
    if not body.get("tools"):
        body.pop("tools", None)

    # remote APIs read their key from the env var named in the config
    headers = {}
    if "api_key_env" in model_config:
        headers["Authorization"] = f"Bearer {os.environ[model_config['api_key_env']]}"

    return requests.post(model_config["url"], json=body, headers=headers)


def invoke_tools(tool_calls, selected_model, models, messages: list) -> list:
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
            response = call_model(selected_model, models, messages)
            response_text = response.json()["choices"][0]["message"]["content"]
            tool_calls = response.json()["choices"][0]["message"].get("tool_calls")
            print(f"response: {response_text}")
        else:
            print(f"unknown or unexpected tool requested: {tool}")

        return messages


def select_model(models):
    names = list(models)
    for i, name in enumerate(names, start=1):
        print(f"{i} : {name}")

    while True:
        choice = input("Select model to use: ")
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print(f"Enter a number between 1 and {len(names)}")
