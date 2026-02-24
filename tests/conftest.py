import os
import sys
from pathlib import Path

# Make src-layout package importable without requiring editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Must be set before any rag_lab import — pydantic-settings reads env vars
# when the Settings class is first instantiated (module-level in config.py).
# setdefault() preserves a real key from .env if present, so tests also
# work in an environment with valid credentials.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-api-key-not-real")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langchain-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
