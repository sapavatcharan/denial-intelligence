"""Prior authorization evidence extractor (CARC 197 / 198)."""

from __future__ import annotations

from ..reference import procedure_info
from ..schemas import Claim, EvidenceItem


def check_prior_auth(claim: Claim) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    sub = claim.submission
    rem = claim.remittance
    if not sub:
        return items

    auth_present = bool((sub.prior_authorization or "").strip()) or bool(
        (rem.prior_auth_num if rem else "") or ""
    )

    proc = claim.primary_procedure
    info = procedure_info(proc) if proc else None

    payer_id = (sub.payer_id or (rem.payer_id if rem else None) or "").upper()
    requires_pa = False
    if info and "requires_prior_auth_payers" in info:
        requires_pa = payer_id in info["requires_prior_auth_payers"]

    if requires_pa and not auth_present:
        items.append(
            EvidenceItem(
                check_name="prior_auth.required_but_missing",
                passed=False,
                severity="critical",
                message=(
                    f"Procedure {proc} requires prior authorization for payer {payer_id}, "
                    "but neither ec_PriorAuthorization nor pc_PriorAuthNum is populated."
                ),
                observed_value=auth_present,
                expected_value=True,
                fields_referenced=["ec_PriorAuthorization", "pc_PriorAuthNum"],
            )
        )
    elif requires_pa and auth_present:
        items.append(
            EvidenceItem(
                check_name="prior_auth.required_and_present",
                passed=True,
                severity="info",
                message=(
                    f"Procedure {proc} requires prior auth for payer {payer_id}; an authorization "
                    "number is present, supporting an appeal if the denial cited missing PA."
                ),
                observed_value=sub.prior_authorization
                or (rem.prior_auth_num if rem else None),
                fields_referenced=["ec_PriorAuthorization", "pc_PriorAuthNum"],
            )
        )
    elif auth_present:
        items.append(
            EvidenceItem(
                check_name="prior_auth.present",
                passed=True,
                severity="info",
                message="A prior authorization number was submitted on the claim.",
                observed_value=sub.prior_authorization
                or (rem.prior_auth_num if rem else None),
                fields_referenced=["ec_PriorAuthorization", "pc_PriorAuthNum"],
            )
        )
    else:
        items.append(
            EvidenceItem(
                check_name="prior_auth.not_required",
                passed=True,
                severity="info",
                message=(
                    f"No PA requirement on file for procedure {proc} with payer {payer_id}; "
                    "any CARC 197/198 denial would warrant payer-policy review."
                ),
                fields_referenced=["ec_PriorAuthorization"],
            )
        )

    return items
