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

Models with "auth_required": true in models.json send the key from the env var named
in "api_key_env" as a Bearer token. Unsloth Studio models use "UNSLOTH_API_KEY", created
in Studio under Settings > API.
"""


def main():
    # load API keys from .env; variables already set in the shell take precedence
    load_dotenv(ENV_PATH)

    # local model configs
    with open(CONFIG_PATH, encoding="utf-8") as f:
        models = json.load(f)

        interaction_mode, selected_model = select_model(models)
        current_model = 0
        initial_prompt = ""
        agent_to_agent_continue = False
        last_agent_to_agent_response = ""

        # agent-to-agent entries list the models to alternate between; single mode
        # is a list of one
        if interaction_mode == "agent-to-agent":
            agent_models = models[selected_model]["models"]
        else:
            agent_models = [selected_model]

        # one conversation history per model, each interaction is appended to
        # maintain conversation history
        # when using 1 model, histories[0] holds the history
        # when using agent to agent, histories[0] is history with model 1,
        # histories[1] is model 2, etc. Each model sees its own replies as
        # "assistant" and the other model's replies as "user"
        histories = [[] for _ in agent_models]

        while True:
            # if running in single model interaction mode, capture prompt from user
            # unless this is the first prompt from the suser for agent to agent
            if interaction_mode == "single" or (
                interaction_mode == "agent-to-agent" and initial_prompt == ""
            ):
                prompt = input("prompt: ")
            else:
                # otherwise the next prompt is the last reply from the other model
                # unless the number of interactions has reached 0
                prompt = last_agent_to_agent_response

            if (
                interaction_mode == "agent-to-agent"
                and models[selected_model].get("interactions") > 0
            ):
                agent_to_agent_continue = True
            else:
                agent_to_agent_continue = False

            if not initial_prompt:
                initial_prompt = prompt

            if prompt not in ("exit", "/exit") and (
                interaction_mode == "single" or agent_to_agent_continue
            ):
                # the model whose turn it is, and its own history
                model_name = agent_models[current_model]
                messages = histories[current_model]
                messages.append({"role": "user", "content": prompt})

                response = call_model(model_name, models, messages)

                print(response.json())
                if response.status_code == 200:
                    response_text = response.json()["choices"][0]["message"]["content"]
                    tool_calls = response.json()["choices"][0]["message"].get(
                        "tool_calls"
                    )
                    print()
                    print(f"response: {response_text}")
                    if tool_calls:
                        messages = invoke_tools(
                            tool_calls, model_name, models, messages
                        )
                    print()

                    # keep the response in context
                    messages.append({"role": "assistant", "content": response_text})

                    if interaction_mode == "agent-to-agent":
                        remaining_interactions = models[selected_model].get(
                            "interactions"
                        )
                        remaining_interactions -= 1
                        models[selected_model]["interactions"] = remaining_interactions

                        # save reply from this model and input to the next
                        last_agent_to_agent_response = response_text
                        # hand the next turn to the other model
                        current_model = (current_model + 1) % len(agent_models)

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

    # models with auth_required read their key from the env var named in api_key_env
    # (loaded from .env) and send it as a Bearer token, as both the Anthropic and
    # Unsloth Studio APIs expect
    headers = {}
    if model_config.get("auth_required"):
        key_env = model_config["api_key_env"]
        api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(
                f"{selected_model} requires an API key: set {key_env} in .env"
            )
        headers["Authorization"] = f"Bearer {api_key}"

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
            model_name = names[int(choice) - 1]
            if model_name.startswith("agent-to-agent"):
                return "agent-to-agent", model_name
            return "single", model_name
        print(f"Enter a number between 1 and {len(names)}")
