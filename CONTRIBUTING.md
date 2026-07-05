# Contributing

Thanks for considering contributing to cn-llm-bridge!

## Setup

```bash
git clone https://github.com/maxliven/cn-llm-bridge.git
cd cn-llm-bridge
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

## Development

### Code Quality

```bash
ruff check . && ruff format .
```

### Running Tests

```bash
pytest tests/ -v
```

> **Note:** Tests use mocked API responses (`respx` for HTTP mocking). No real API keys are needed to run the test suite.

### Project Structure

```
cn-llm-bridge/
├── cn_llm_bridge/        # Qwen vision + faster-whisper MCP server
│   ├── __init__.py
│   └── server.py
├── kimi_bridge/          # Kimi K2.7 Code MCP server
│   ├── __init__.py
│   └── server.py
├── tests/                # Test suite
├── docs/                 # Documentation
├── pyproject.toml
├── .env.example
└── README.md
```

## Pull Request Guidelines

- Write a clear description of the change and why it's needed
- Link to any related issues
- Add or update tests for the changed behavior
- Run `ruff check . && ruff format .` before committing
- Run `pytest tests/ -v` and ensure all pass
- **Never commit `.env` files or API keys**
- If you add a new environment variable, update `.env.example`

## Adding a New Model Bridge

1. Create a new package directory (e.g., `glm_bridge/`)
2. Add `__init__.py` with version and docstring
3. Add `server.py` implementing the MCP server pattern
4. Register in `pyproject.toml` under `[project.scripts]` and `[tool.hatch.build.targets.wheel]`
5. Add tests and documentation
