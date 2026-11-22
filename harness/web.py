"""Web chat frontend for the harness.

The user enters an initial prompt, which is sent to the first model; its reply is sent
to the second model, and so on, alternating for the chosen number of turns. Models are
called one at a time, in sequence.

The browser keeps each model's conversation history and calls /api/chat once per turn;
this server only forwards each call to the model API via call_model, so API keys from
.env never reach the browser. Tools are not sent to the models from the web UI, since
there is no way to confirm a bash command before it runs.

Run with `poetry run harness-web`, then open http://127.0.0.1:8765
"""

import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from harness.main import CONFIG_PATH, ENV_PATH, call_model

STATIC_DIR = Path(__file__).parent / "static"

# localhost only: anyone who can reach this server can spend your API credits
HOST = "127.0.0.1"
PORT = 8765

app = Flask(__name__, static_folder=None)


def load_models():
    # re-read on each request so edits to models.json apply without a restart
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def agent_to_agent_system_prompt(models):
    # the system_prompt from the agent-to-agent entry in models.json, if set
    for name, config in models.items():
        if name.startswith("agent-to-agent"):
            return config.get("system_prompt")
    return None


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/models")
def list_models():
    models = load_models()
    # only actual models, leaving out the agent-to-agent entry
    return jsonify([name for name in models if "message_body" in models[name]])


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    messages = data.get("messages")

    models = load_models()
    if model_name not in models or "message_body" not in models[model_name]:
        return jsonify(error=f"unknown model: {model_name}"), 400
    if not isinstance(messages, list):
        return jsonify(error="messages must be a list"), 400

    # every web conversation is agent-to-agent, so lead with its system prompt
    system_prompt = agent_to_agent_system_prompt(models)
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    try:
        response = call_model(model_name, models, messages, use_tools=False)
    except SystemExit as e:
        # call_model exits when a required API key isn't set
        return jsonify(error=str(e)), 400
    except requests.RequestException as e:
        return jsonify(error=f"could not reach {model_name}: {e}"), 502

    if response.status_code != 200:
        return jsonify(
            error=f"{model_name} returned {response.status_code}: {response.text}"
        ), 502

    reply = response.json()["choices"][0]["message"].get("content") or ""
    return jsonify(reply=reply)


def run():
    # load API keys from .env; variables already set in the shell take precedence
    load_dotenv(ENV_PATH)
    print(f"Harness web UI: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT)
