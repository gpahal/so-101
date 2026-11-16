"""API keys for the vision judges, from a KEY=value file such as `.scratch/bench.env` beside so101.toml."""

import os
from pathlib import Path


def load_env(path: Path) -> list[str]:
    """Read KEY=value lines, or `export KEY=value` as a shell would source them, into the environment,
    without overriding variables already set. Returns the names it set (never the values)."""
    if not path.exists():
        return []
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if value and not os.environ.get(name):
            os.environ[name] = value
            names.append(name)
    return names
