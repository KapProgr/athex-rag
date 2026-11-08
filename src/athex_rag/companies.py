"""Company metadata: display names and the aliases people use in questions.

Lives in ``data/companies.json`` (override with ``ATHEX_COMPANIES``). Greek names are
inflected ("η Εθνική Τράπεζα", "της Εθνικής Τράπεζας"), so list the forms you expect.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PATH = Path("data/companies.json")


def load_companies(path: Path | None = None) -> dict[str, dict]:
    path = path or Path(os.environ.get("ATHEX_COMPANIES", DEFAULT_PATH))
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    return {code: info.get("aliases", []) for code, info in load_companies(path).items()}


def display_name(code: str, companies: dict[str, dict]) -> str:
    return companies.get(code, {}).get("name", code)
