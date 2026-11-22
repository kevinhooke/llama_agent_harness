# Simple llama.cpp ai harness

Experiemental investigation around building an agent harness for LLM models
hosted on a locally running llama.cpp server.

## How to configure your models

Configure the `./config/models.json` file to configure the models you have available.

Currently supports calling models using the OpenAI API, which includes (and has been tested with):

- Local llama.cpp server (no auth or key required)
- Local Unsloth Studio server (requires an unsloth api key)
- Anthropic cloud models (requires an Anthropic API key)

## How to run
Install before running for first time:
```
poetry install
```

To run:
```
poetry run harness
```

## Single model chat

On starting up you are prompted to select a model to interact with, pick
from your configured list from `./config/models.json` :
```
❯ poetry run harness
1 : qwen3.6-35b-a3b
2 : unsloth-Tiel-Coder-35B-A3B-GGUF
3 : opus-5.5
4 : agent-to-agent
Select model to use:
```

## Model to model chat

If you pick option 4 (model-to-model), you are prompted to pick two
models that are used in an agent to agent converstation. You start the
conversation with the first promp that is sent to agent 1. The reply is then
sent to agent 2.

This doesn't serve any practical purpose, it is useful to explore the
possibility of how agents can work together con colloborative tasks, or
to facilitate tasks split between multiple agents.

## LLM Chat Room

If you start with this option:
```
poetry run harness-web
```
a web based app is started that displays the agent to agent chats on the page.

To customise how the agents respond in the chat, try changing the system prompt
in `./config/models.json` for the 'agent-to-agent' config. It's currently configured
for a general conversational chat:

```
"You are a participant in an agent to agent chat session. Any reply is sent to
the other participant agents, who will reply to you. Talk about random topics
to pass the time of day."
```

## Observations

- sending a single `{"role": "user", "content": prompt}` message to a model does not
maintain conversational history. In order to ask the model followup quesions about
what has been discussed so far, you need to append each successive chat interaction
to an array of messages sent in the api call

- the purpose of `"role"` describes whether the request is from the `user` or a response from the model (`assistant`)

## References

- chat completions api docs: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create?site_locale=en
