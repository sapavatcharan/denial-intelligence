"""Loaders for the reference data files in `data/reference/`."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reference"


def _load(filename: str) -> dict[str, Any]:
    path = REFERENCE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def carc_codes() -> dict[str, dict[str, Any]]:
    data = _load("carc_codes.json")
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def rarc_codes() -> dict[str, dict[str, Any]]:
    data = _load("rarc_codes.json")
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def payer_filing_limits() -> dict[str, Any]:
    return _load("payer_filing_limits.json")


@lru_cache(maxsize=1)
def dx_procedure_pairings() -> dict[str, Any]:
    return _load("dx_procedure_pairings.json")


def carc(code: str) -> dict[str, Any] | None:
    return carc_codes().get(code)


def rarc(code: str) -> dict[str, Any] | None:
    return rarc_codes().get(code)


def filing_limit_days(payer_id: str | None, insurance_type: str | None) -> tuple[int, str]:
    """Return (days, source_label) for the given payer / insurance type."""
    table = payer_filing_limits()
    if payer_id and payer_id.upper() in table["by_payer_id"]:
        entry = table["by_payer_id"][payer_id.upper()]
        return int(entry["days"]), f"payer:{entry['name']}"
    if insurance_type and insurance_type in table["by_insurance_type_default"]:
        return int(table["by_insurance_type_default"][insurance_type]), f"insurance_type:{insurance_type}"
    return int(table["default_days"]), "default"


def procedure_info(procedure_code: str | None) -> dict[str, Any] | None:
    if not procedure_code:
        return None
    return dx_procedure_pairings()["procedures"].get(procedure_code)
