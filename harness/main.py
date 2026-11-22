import json
import os
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests import Response

CONFIG_PATH = Path(__file__).parent / "config" / "models.json"
SETTINGS_PATH = Path(__file__).parent / "config" / "settings.json"
ENV_PATH = Path(__file__).parent.parent / ".env"

# harness settings, e.g. dump_api_payloads to print request and response JSON
with open(SETTINGS_PATH, encoding="utf-8") as f:
    SETTINGS = json.load(f)

"""Simple harness for interacting with LLMs via the Anthropic API style messages.

Available models are configured in config/models.json. Configure and update as needed.

Harness settings are in config/settings.json. Set "dump_api_payloads" to true to print
the JSON sent to and received from the model APIs.

Run with `poetry run harness`.

At start up, prompts user to select which model to interact with. Selecting
agent-to-agent prompts for two of the configured models, then after the user's first
prompt the two models converse with each other for the configured number of
interactions.

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

        # agent_models holds the models to alternate between: the two the user
        # picked for agent-to-agent, or a list of one in single mode
        interaction_mode, selected_model, agent_models = select_model(models)
        current_model = 0
        initial_prompt = ""
        agent_to_agent_continue = False
        last_agent_to_agent_response = ""

        # one conversation history per model, each interaction is appended to
        # maintain conversation history
        # when using 1 model, histories[0] holds the history
        # when using agent to agent, histories[0] is history with model 1,
        # histories[1] is model 2, etc. Each model sees its own replies as
        # "assistant" and the other model's replies as "user"
        histories = [[] for _ in agent_models]

        # in agent-to-agent mode, each model's history starts with the configured
        # system prompt telling it that its replies go to the other participant
        if interaction_mode == "agent-to-agent":
            system_prompt = models[selected_model].get("system_prompt")
            if system_prompt:
                for history in histories:
                    history.append({"role": "system", "content": system_prompt})

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

                if response.status_code == 200:
                    response_text = response.json()["choices"][0]["message"]["content"]
                    tool_calls = response.json()["choices"][0]["message"].get(
                        "tool_calls"
                    )
                    print()
                    print(f"[{model_name}]: {response_text}")
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


def call_model(selected_model, models, messages, use_tools=True) -> Response:

    model_config = models[selected_model]

    # TODO track messages state outside later
    # copy the configured body so the loaded config isn't modified
    body = dict(model_config["message_body"], messages=messages)

    # an empty tools list can be rejected by the API, so only send it when populated.
    # callers that can't run tools (e.g. the web UI) pass use_tools=False to leave
    # them out entirely
    if not use_tools or not body.get("tools"):
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

    if SETTINGS.get("dump_api_payloads"):
        print(f"request payload: {json.dumps(body, indent=2)}")

    response = requests.post(model_config["url"], json=body, headers=headers)

    # response.text rather than .json() so non-JSON error bodies can still be dumped
    if SETTINGS.get("dump_api_payloads"):
        print(f"response payload: {response.text}")

    return response


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
            print(f"[{selected_model}]: {response_text}")
        else:
            print(f"unknown or unexpected tool requested: {tool}")

        return messages


def select_model(models):
    model_name = choose(list(models), "Select model to use: ")
    if not model_name.startswith("agent-to-agent"):
        return "single", model_name, [model_name]

    # agent-to-agent: pick the two models to converse from the configured models,
    # leaving out the agent-to-agent entry itself
    chat_models = [name for name in models if "message_body" in models[name]]
    print()
    model_1 = choose(chat_models, "Select model 1: ")
    print()
    model_2 = choose(chat_models, "Select model 2: ")
    return "agent-to-agent", model_name, [model_1, model_2]


def choose(names, prompt):
    # print a numbered menu and return the name the user picks
    for i, name in enumerate(names, start=1):
        print(f"{i} : {name}")

    while True:
        choice = input(prompt)
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print(f"Enter a number between 1 and {len(names)}")
