"""Missing-info evidence extractor (CARC 16 / 252).

Driven by the RARC codes attached to the line. Each RARC has a
`missing_field_hint` mapping in `data/reference/rarc_codes.json`; this
extractor checks whether those fields are populated on the claim and
reports specifically which one is missing.
"""

from __future__ import annotations

from typing import Any

from ..reference import rarc
from ..schemas import Claim, EvidenceItem


def _value_at(claim: Claim, field_name: str) -> Any:
    alias_map = {
        "ec_PriorAuthorization": claim.submission.prior_authorization
        if claim.submission
        else None,
        "ec_OtherPayerName": claim.submission.other_payer_name if claim.submission else None,
        "ec_OtherPayerPaid": claim.submission.other_payer_paid if claim.submission else None,
        "ec_PrincipalDiagnosis": claim.submission.principal_diagnosis
        if claim.submission
        else None,
        "pcl_ProcedureCode": claim.primary_procedure,
        "pcl_ChargedAmount": claim.remittance.lines[0].charged_amount
        if claim.remittance and claim.remittance.lines
        else None,
        "pcl_ProcedureModifier1": (
            claim.remittance.lines[0].modifiers[0]
            if claim.remittance and claim.remittance.lines and claim.remittance.lines[0].modifiers
            else None
        ),
        "pcl_ProcedureModifier2": (
            claim.remittance.lines[0].modifiers[1]
            if claim.remittance
            and claim.remittance.lines
            and len(claim.remittance.lines[0].modifiers) >= 2
            else None
        ),
        "pcl_SubmittedProcedureCodeDesc": None,
    }
    return alias_map.get(field_name)


def check_missing_info(claim: Claim) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    rem = claim.remittance
    if not rem:
        return items

    seen_remarks: set[str] = set()
    for line in rem.lines:
        for code in line.remark_codes:
            if code in seen_remarks:
                continue
            seen_remarks.add(code)
            ref = rarc(code)
            if not ref:
                continue
            hints = ref.get("missing_field_hint", [])
            if not hints:
                continue
            missing: list[str] = []
            for fname in hints:
                v = _value_at(claim, fname)
                if v is None or v == "" or v == 0:
                    missing.append(fname)

            passed = not missing
            items.append(
                EvidenceItem(
                    check_name=f"missing_info.rarc_{code}",
                    passed=passed,
                    severity="critical" if not passed else "info",
                    message=(
                        f"RARC {code}: {ref['description']} "
                        + (
                            f"Missing field(s): {missing}."
                            if missing
                            else "All hinted fields are populated."
                        )
                    ),
                    observed_value=",".join(missing) if missing else "all_present",
                    fields_referenced=hints,
                )
            )

    return items
