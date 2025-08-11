# Repository Guidelines

This project is a Flask-based chatbot that integrates OpenAI via LangChain, FAISS for retrieval, and a simple web UI. Use this guide to contribute efficiently and consistently.

## Project Structure & Module Organization
- `app.py`: Main Flask app, routes, LangChain/FAISS wiring.
- `templates/` and `static/`: Jinja templates and assets for the UI.
- `tests/`: Pytest suite (files named `test_*.py`).
- `docs/`: Architectural/technical docs.
- Runtime data: `data/` (FAISS index, caches) and `uploads/` are created on demand.

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt`
- Run dev server: `python app.py` (binds to port 5000)
- Lint: `flake8`
- Format (check): `black --check .` (format with `black .`)
- Tests: `pytest -q`
- Docker: `docker build -t docuchat .` then `docker run -p 5000:5000 -e OPENAI_API_KEY=... -e SECRET_KEY=... docuchat`

## Coding Style & Naming Conventions
- Formatter: `black` with line length 120.
- Linting: `flake8` (see `.flake8` for rules). Remove unused imports; keep imports sorted.
- Docstrings: Add clear, concise docstrings to new functions/classes.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- Framework: `pytest` with tests under `tests/` named `test_*.py`.
- Write unit tests for new routes, helpers, and error paths.
- Run `pytest -q` locally; avoid network calls in tests (use `monkeypatch` as in existing tests).

## Commit & Pull Request Guidelines
- Commits: Use clear, imperative messages (e.g., "Add upload validation"). Group related changes.
- PRs: Describe the change, rationale, and touchpoints (files/lines). Include results of `black --check .`, `flake8`, and `pytest -q`.
- Scope: Keep PRs focused. Include screenshots for UI changes when relevant.

## Security & Configuration
- Required env vars: `OPENAI_API_KEY`, `SECRET_KEY` (use a `.env` locally; never commit secrets).
- Be mindful of logging; avoid printing sensitive data. Large files are limited to 16MB by default.
