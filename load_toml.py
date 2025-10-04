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
from typing import Dict, Any

_HAS_TOMLIB = False
_TOML_LOADER = None
try:
    import tomllib as _tomllib  # Python 3.11+
    _HAS_TOMLIB = True
    _TOML_LOADER = lambda f: _tomllib.load(f)
except Exception:
    try:
        import toml as _toml  # type: ignore
        _TOML_LOADER = lambda f: _toml.loads(f.read().decode() if hasattr(f, 'read') else f)
    except Exception:
        _TOML_LOADER = None


def _set_env_from_mapping(mapping: Dict[str, Any]) -> None:
    """Set environment variables from a mapping, without overwriting existing vars."""
    if not mapping:
        return
    for k, v in mapping.items():
        # Only set if not already present in the environment to allow overrides
        if k not in os.environ and v is not None:
            os.environ[str(k)] = str(v)


def _load_toml_file(path: pathlib.Path) -> Dict[str, Any]:
    if not _TOML_LOADER:
        raise RuntimeError("No TOML loader available. Please run on Python 3.11+ or install the 'toml' package.")
    with path.open("rb") as f:
        return _TOML_LOADER(f)


def load_toml_env(path: str = None) -> None:
    """Load secrets into os.environ.

    Priority:
      1. If running inside Streamlit and `st.secrets` is available, use it.
      2. `.streamlit/secrets.toml` if present in the project root.
      3. `config.toml` next to this loader (project root).

    This function mirrors Streamlit's recommended secrets approach while
    also exporting values to `os.environ` so existing code using `os.getenv`
    continues to work.
    """

    # 1) Streamlit runtime secrets (highest priority)
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and st.secrets:
            # st.secrets may contain nested tables. We support common shapes:
            # - {"GOOGLE_API_KEY": "...", ...}
            # - {"secrets": {...}, "settings": {...}}
            data: Dict[str, Any] = {}
            if "secrets" in st.secrets and isinstance(st.secrets["secrets"], dict):
                data.update(st.secrets["secrets"])
            if "settings" in st.secrets and isinstance(st.secrets["settings"], dict):
                data.update(st.secrets["settings"])
            # Flatten top-level dict entries
            for k, v in st.secrets.items():
                if k in ("secrets", "settings"):
                    continue
                # If value is a mapping, merge its items; otherwise set directly
                if isinstance(v, dict):
                    for ik, iv in v.items():
                        data.setdefault(ik, iv)
                else:
                    data.setdefault(k, v)

            _set_env_from_mapping(data)
            return
    except Exception:
        # Not running under Streamlit or st not available — continue
        pass

    # Helper to find project root (dir containing this file)
    base_dir = pathlib.Path(path or os.path.join(os.path.dirname(__file__), "config.toml")).resolve().parent

    # 2) Check for .streamlit/secrets.toml in project root
    streamlit_secrets = base_dir / ".streamlit" / "secrets.toml"
    if streamlit_secrets.exists() and _TOML_LOADER:
        try:
            data = _load_toml_file(streamlit_secrets)
            # Streamlit secrets file often has top-level keys or tables; merge
            # Use 'secrets' and 'settings' tables if present, otherwise merge top-level
            for table in ("secrets", "settings"):
                tbl = data.get(table)
                if isinstance(tbl, dict):
                    _set_env_from_mapping(tbl)
            # top-level
            _set_env_from_mapping({k: v for k, v in data.items() if not isinstance(v, dict)})
            return
        except Exception:
            pass

    # 3) Fallback to config.toml next to this file
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.toml")

    p = pathlib.Path(path)
    if not p.exists():
        # Nothing to load; keep environment as-is
        return

    data = _load_toml_file(p)
    # Merge 'secrets' and 'settings' tables if present
    for table in ("secrets", "settings"):
        tbl = data.get(table)
        if isinstance(tbl, dict):
            _set_env_from_mapping(tbl)



if __name__ == "__main__":
    load_toml_env()
