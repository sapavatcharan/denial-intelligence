"""Medical necessity evidence extractor (CARC 11 / 50 / 167).

Compares the procedure billed against the curated diagnosis pairings and
reports the strongest supporting diagnosis on the claim - including
secondary diagnoses, which is the most common appealable scenario.
"""

from __future__ import annotations

from ..reference import procedure_info
from ..schemas import Claim, EvidenceItem


def _matches(dx: str, prefixes: list[str]) -> bool:
    if "any" in prefixes:
        return True
    return any(dx.upper().startswith(p.upper()) for p in prefixes)


def check_medical_necessity(claim: Claim) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    proc = claim.primary_procedure
    info = procedure_info(proc)
    if not info or "supportive_dx_prefixes" not in info:
        return items

    prefixes = list(info["supportive_dx_prefixes"])
    diagnoses = claim.all_diagnoses
    if not diagnoses:
        items.append(
            EvidenceItem(
                check_name="medical_necessity.no_diagnoses",
                passed=False,
                severity="critical",
                message="No diagnoses present on the 837; medical-necessity assessment not possible.",
                fields_referenced=["ec_PrincipalDiagnosis"],
            )
        )
        return items

    principal = diagnoses[0]
    principal_supports = _matches(principal, prefixes)

    secondary_supports: list[str] = []
    for dx in diagnoses[1:]:
        if _matches(dx, prefixes):
            secondary_supports.append(dx)

    if principal_supports:
        items.append(
            EvidenceItem(
                check_name="medical_necessity.principal_supports",
                passed=True,
                severity="info",
                message=(
                    f"Principal diagnosis {principal} is on the supportive list "
                    f"({prefixes}) for procedure {proc}; medical necessity is established."
                ),
                observed_value=principal,
                fields_referenced=["ec_PrincipalDiagnosis"],
            )
        )
    elif secondary_supports:
        items.append(
            EvidenceItem(
                check_name="medical_necessity.secondary_supports",
                passed=True,
                severity="warning",
                message=(
                    f"Principal diagnosis {principal} does not match the supportive list, but "
                    f"secondary diagnosis(es) {secondary_supports} do. Repointing the diagnosis "
                    "pointer is a strong appeal path."
                ),
                observed_value=",".join(secondary_supports),
                fields_referenced=[
                    "ec_PrincipalDiagnosis",
                    "ec_Diag2",
                    "ec_Diag3",
                    "ec_Diag4",
                ],
            )
        )
    else:
        items.append(
            EvidenceItem(
                check_name="medical_necessity.no_dx_supports",
                passed=False,
                severity="critical",
                message=(
                    f"None of the submitted diagnoses {diagnoses} match the supportive list "
                    f"{prefixes} for procedure {proc}. Appeal will require additional clinical documentation."
                ),
                observed_value=",".join(diagnoses),
                expected_value=",".join(prefixes),
                fields_referenced=["ec_PrincipalDiagnosis", "ec_Diag2"],
            )
        )

    return items
