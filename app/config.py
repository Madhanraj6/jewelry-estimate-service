"""Load environment configuration from a local .env file, if one exists.

Imported first by ``app/__init__.py`` so that every entry point - the API, the
evaluation harness, and the two maintenance scripts - picks up the same settings before
any module reads them.

``.env`` is already in ``.gitignore``. Real environment variables always win: a value
exported in the shell or injected by an orchestrator takes precedence over the file, so
the same image behaves correctly in Docker and on a developer machine.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env() -> Path | None:
    """Load ``.env`` if present. Returns the path loaded, or None."""
    if not ENV_FILE.is_file():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Not fatal: the app runs on environment variables alone.
        return None
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    return ENV_FILE


def api_key_configured() -> bool:
    """Report whether any LLM credential is present, without printing it."""
    return bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


load_env()