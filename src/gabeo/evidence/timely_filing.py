"""Timely filing evidence extractor (CARC 29).

Computes the actual days between service date and the payer's receipt of
the claim, looks up the payer's filing window, and produces an evidence
item that the LLM cannot disagree with on the math.

Handles the secondary-payer case: when the 837 includes an
`ec_OtherPayerPaidDate`, the filing window for the secondary submission
typically anchors to the primary EOB date, not the original date of service.
"""

from __future__ import annotations

from datetime import date

from ..reference import filing_limit_days
from ..schemas import Claim, EvidenceItem


def _days_between(a: date, b: date) -> int:
    return (b - a).days


def check_timely_filing(claim: Claim) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    sub = claim.submission
    rem = claim.remittance
    if not sub or not rem:
        return items

    service_date = sub.service_date_from
    received_date = rem.received_date
    if not service_date or not received_date:
        items.append(
            EvidenceItem(
                check_name="timely_filing.dates_present",
                passed=False,
                severity="warning",
                message="Cannot evaluate timely filing without both ec_ServiceDateFrom and pc_ReceivedDate.",
                fields_referenced=["ec_ServiceDateFrom", "pc_ReceivedDate"],
            )
        )
        return items

    payer_id = sub.payer_id or rem.payer_id
    insurance_type = sub.insurance_type or rem.insurance_type
    limit_days, source = filing_limit_days(payer_id, insurance_type)

    is_secondary = sub.other_payer_paid_date is not None
    anchor_date: date = service_date
    anchor_label = "ec_ServiceDateFrom"
    if is_secondary and sub.other_payer_paid_date is not None:
        anchor_date = sub.other_payer_paid_date
        anchor_label = "ec_OtherPayerPaidDate"

    days_elapsed = _days_between(anchor_date, received_date)
    within_limit = days_elapsed <= limit_days

    items.append(
        EvidenceItem(
            check_name="timely_filing.window_check",
            passed=within_limit,
            severity="critical" if not within_limit else "info",
            message=(
                f"Claim received {days_elapsed} day(s) after {anchor_label}; "
                f"payer filing window is {limit_days} day(s) "
                f"({'WITHIN' if within_limit else 'OVER'} the limit; source: {source})."
            ),
            observed_value=days_elapsed,
            expected_value=limit_days,
            fields_referenced=[anchor_label, "pc_ReceivedDate"],
        )
    )

    if sub.delay_reason_code:
        items.append(
            EvidenceItem(
                check_name="timely_filing.delay_reason_present",
                passed=True,
                severity="info",
                message=(
                    f"A delay reason code is present (ec_DelayReasonCode={sub.delay_reason_code}). "
                    "This often supports an appeal even when the window is exceeded."
                ),
                observed_value=sub.delay_reason_code,
                fields_referenced=["ec_DelayReasonCode"],
            )
        )

    if is_secondary:
        items.append(
            EvidenceItem(
                check_name="timely_filing.secondary_anchor",
                passed=True,
                severity="info",
                message=(
                    "Claim is secondary; filing window anchored to ec_OtherPayerPaidDate. "
                    "If the payer used ec_ServiceDateFrom instead, the denial is appealable."
                ),
                observed_value=str(sub.other_payer_paid_date),
                fields_referenced=["ec_OtherPayerPaidDate", "ec_ServiceDateFrom"],
            )
        )

    return items
