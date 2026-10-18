# Simple llama.cpp ai harness

Experiemental investigation around building an agent harness for LLM models
hosted on a locally running llama.cpp server.

## Observations

- sending a single `{"role": "user", "content": prompt}` message to a model does not
maintain conversational history. In order to ask the model followup quesions about
what has been discussed so far, you need to append each successive chat interaction
to an array of messages sent in the api call
