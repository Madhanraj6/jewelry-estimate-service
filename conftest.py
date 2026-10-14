"""Make the project root importable so `pytest` works without setting PYTHONPATH."""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests should never consume configured provider quota or compete with a running API
# process for Qdrant's single-process local-storage lock.
for variable in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ[variable] = ""
os.environ["IMAGE_EMBEDDING_BACKEND"] = "opencv"
os.environ["INDEX_EMBEDDING_BACKEND"] = "opencv"
_TEST_INDEX_DIR = Path(tempfile.mkdtemp(prefix="jewelry-estimate-qdrant-tests-"))
os.environ["CATALOG_INDEX_PATH"] = str(_TEST_INDEX_DIR)
os.environ["CATALOG_DESCRIPTION_CACHE"] = str(_TEST_INDEX_DIR / "descriptions.json")


def _cleanup_test_index() -> None:
    catalog = sys.modules.get("app.catalog_index")
    if catalog is not None:
        catalog._close_clients()
    shutil.rmtree(_TEST_INDEX_DIR, ignore_errors=True)


atexit.register(_cleanup_test_index)