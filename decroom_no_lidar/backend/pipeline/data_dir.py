"""
Private on-disk location for backend scratch + debug files.

Replaces the old predictable /tmp/decroom_* paths: /tmp is world-writable,
so another local user could pre-plant a symlink there and redirect our
writes (or the 7-day keyframe prune's rmtree). Everything now lives under
DECROOM_DATA_DIR (default ~/.decroom/data), created with mode 0700.
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Return the private data dir, creating it (mode 0700) if needed."""
    default = Path.home() / ".decroom" / "data"
    root = Path(os.environ.get("DECROOM_DATA_DIR", str(default))).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    # mkdir's mode is masked by umask and ignored for existing dirs.
    os.chmod(root, 0o700)
    if root == default:
        os.chmod(root.parent, 0o700)
    return root
