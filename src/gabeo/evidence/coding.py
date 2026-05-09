"""Coding / modifier evidence extractor (CARC 4)."""

from __future__ import annotations

from ..reference import procedure_info
from ..schemas import Claim, EvidenceItem


def check_coding(claim: Claim) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    rem = claim.remittance
    if not rem or not rem.lines:
        return items

    for idx, line in enumerate(rem.lines):
        proc = line.procedure_code
        info = procedure_info(proc)
        if not info:
            continue
        required = set(info.get("common_required_modifiers", []))
        present = set(line.modifiers)

        if not required:
            continue

        paired = {"LT", "RT", "50"}
        if required <= paired:
            satisfied = bool(present & paired)
        else:
            satisfied = required.issubset(present)

        items.append(
            EvidenceItem(
                check_name=f"coding.line_{idx}.required_modifier",
                passed=satisfied,
                severity="critical" if not satisfied else "info",
                message=(
                    f"Procedure {proc} commonly requires modifiers {sorted(required)}; "
                    f"submitted modifiers: {sorted(present) or '(none)'} - "
                    f"{'OK' if satisfied else 'MISMATCH'}."
                ),
                observed_value=",".join(sorted(present)) if present else "",
                expected_value=",".join(sorted(required)),
                fields_referenced=[
                    "pcl_ProcedureCode",
                    "pcl_ProcedureModifier1",
                    "pcl_ProcedureModifier2",
                    "pcl_ProcedureModifier3",
                    "pcl_ProcedureModifier4",
                ],
            )
        )

    return items
