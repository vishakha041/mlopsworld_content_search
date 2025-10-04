"""
Small helper to load environment variables from config.toml into os.environ.

This is intentionally minimal: it reads the `secrets` and `settings` tables
from `config.toml` and exports their keys into the process environment so
existing code that uses `os.getenv` continues to work without other changes.

Requires Python 3.11+ (tomllib). If you need older Python support, we can
switch to the third-party `toml` package instead.
"""
import os
import pathlib

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None


def load_toml_env(path: str = None) -> None:
    """Load config.toml and set os.environ values.

    Args:
        path: Optional path to the toml file. Defaults to ./config.toml
    """
    if tomllib is None:
        # Fail fast with a clear message
        raise RuntimeError("tomllib not available. Please run on Python 3.11+ or install 'toml' and update this loader.")

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.toml")

    p = pathlib.Path(path)
    if not p.exists():
        # Nothing to load; keep environment as-is
        return

    with p.open("rb") as f:
        data = tomllib.load(f)

    # Merge 'secrets' and 'settings' tables if present
    for table in ("secrets", "settings"):
        tbl = data.get(table)
        if not isinstance(tbl, dict):
            continue
        for k, v in tbl.items():
            # Only set if not already present in the environment to allow overrides
            if k not in os.environ:
                os.environ[str(k)] = str(v)


if __name__ == "__main__":
    load_toml_env()
