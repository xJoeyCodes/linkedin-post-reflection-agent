# Reflection Agent

Side Note: For most of my Agentic projects I use IBM Watson and Watsonx models, because....... drumroll........ I have a free trial using their model so I am making use of it and building away (for now)!! 

This repository contains a small agent that uses an LLM to generate and reflect on LinkedIn posts.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your model provider credentials. Example:

```bash
cp .env.example .env
# edit .env and add real API keys
```

3. Run the agent:

```bash
python agent.py
```

Security

- Never commit your `.env` file. `.gitignore` already excludes it.
- Use environment variables or secret stores for API keys.

Contributing

Open an issue or send a PR.
