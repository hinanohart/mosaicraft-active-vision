"""Test-time bootstrap so ``mosaicraft`` (git submodule) is importable.

The submodule is intentionally not pip-installed (it pulls in
``opencv-python`` while this repo uses ``opencv-python-headless``); the
package code injects ``external/mosaicraft/src`` into ``sys.path`` at
import time, but pytest may import test files before any package module
is touched. This conftest does the same injection up-front so tests that
import mosaicraft sub-symbols directly (e.g. for golden-hash fixtures)
work the same way the runtime does.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOSAICRAFT_SRC = _REPO_ROOT / "external" / "mosaicraft" / "src"
if _MOSAICRAFT_SRC.is_dir():
    sys.path.insert(0, str(_MOSAICRAFT_SRC))
