#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SUFFIX = ".before-chat-timestamps.bak"
FILES = [
    Path("backend/app/models.py"),
    Path("backend/app/api.py"),
    Path("frontend/src/services/api.ts"),
    Path("frontend/src/App.vue"),
    Path("frontend/src/components/ChatPanel.vue"),
]

parser = argparse.ArgumentParser()
parser.add_argument("repository", type=Path)
args = parser.parse_args()
root = args.repository.expanduser().resolve()
restored = 0
for relative in FILES:
    path = root / relative
    backup = path.with_name(path.name + SUFFIX)
    if backup.exists():
        shutil.copy2(backup, path)
        backup.unlink()
        print(f"Restored {relative}")
        restored += 1
print(f"Restored {restored} file(s).")
