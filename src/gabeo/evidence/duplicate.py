"""Duplicate evidence extractor (CARC 18 / 97).

Without a historical store this module reports only structural signals from
the claim itself (e.g., bilateral procedure with no LT/RT modifier - a
common cause of a CARC 18 mis-denial). The retrieval layer adds the
"have we seen this exact (patient, dos, procedure) before?" lookup later.
"""

from __future__ import annotations

from ..reference import procedure_info
from ..schemas import Claim, EvidenceItem


def check_duplicate(claim: Claim) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    rem = claim.remittance
    if not rem or not rem.lines:
        return items

    for idx, line in enumerate(rem.lines):
        proc = line.procedure_code
        info = procedure_info(proc)
        if not info:
            continue
        paired = {"LT", "RT", "50"}
        common_required = set(info.get("common_required_modifiers", []))
        if not (common_required & paired):
            continue
        present = set(line.modifiers) & paired
        if not present:
            items.append(
                EvidenceItem(
                    check_name=f"duplicate.line_{idx}.bilateral_modifier_missing",
                    passed=False,
                    severity="warning",
                    message=(
                        f"Procedure {proc} is a paired-organ procedure; without LT/RT/50 modifiers, "
                        "payers commonly mis-flag a legitimate second-side service as a duplicate. "
                        "Resubmit with the appropriate side modifier."
                    ),
                    fields_referenced=[
                        "pcl_ProcedureCode",
                        "pcl_ProcedureModifier1",
                        "pcl_ProcedureModifier2",
                    ],
                )
            )
        else:
            items.append(
                EvidenceItem(
                    check_name=f"duplicate.line_{idx}.bilateral_modifier_present",
                    passed=True,
                    severity="info",
                    message=(
                        f"Procedure {proc} has side modifier(s) {sorted(present)}; "
                        "this argues against a true-duplicate denial."
                    ),
                    observed_value=",".join(sorted(present)),
                    fields_referenced=[
                        "pcl_ProcedureCode",
                        "pcl_ProcedureModifier1",
                    ],
                )
            )

    return items
