from __future__ import annotations

from pathlib import Path
import sys


def _add_to_syspath(path: Path) -> None:
    if not path.is_dir():
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


ROOT = Path(__file__).resolve().parent

# Make local workspace packages importable without installing them.
_add_to_syspath(ROOT / "livekit-agents")

plugins_root = ROOT / "livekit-plugins"
if plugins_root.is_dir():
    for plugin_dir in plugins_root.iterdir():
        if plugin_dir.is_dir() and (plugin_dir / "livekit").is_dir():
            _add_to_syspath(plugin_dir)
