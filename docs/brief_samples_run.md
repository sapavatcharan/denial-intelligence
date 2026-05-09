# Brief Sample Verdicts (live LLM run)

These are the unedited verdicts produced by `gabeo analyze` on the four sample claims from the assignment PDF, plus the deterministic evidence that grounded each verdict.

## CLM-2026-00142

- **Recoverability:** `not_recoverable`
- **Confidence:** 0.60 (components: {'llm': 0.7, 'evidence_pass_rate': 0.5, 'blended': 0.6})
- **Model:** mock:gpt-4o | cost: $0.00000 | latency: 6 ms

**Root cause.** Denial driven by CARC 29. Claim received 278 day(s) after ec_ServiceDateFrom; payer filing window is 180 day(s) (OVER the limit; source: payer:BCBS Illinois). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate']) Procedure 99214 commonly requires modifiers ['25']; submitted modifiers: (none) - MISMATCH. (fields_referenced=['pcl_ProcedureCode', 'pcl_ProcedureModifier1', 'pcl_ProcedureModifier2', 'pcl_ProcedureModifier3', 'pcl_ProcedureModifier4'])

**CARC interpretation.** CARC 29 (timely_filing): the payer adjudicated this charge under the timely_filing family of denials.

**Recommended action.** Write off and document the denial reason in the contractual-adjustment log.

**Supporting evidence (cited fields):**

- `ec_ServiceDateFrom` = `2025-06-15` — Claim received 278 day(s) after ec_ServiceDateFrom; payer filing window is 180 day(s) (OVER the limit; source: payer:BCBS Illinois). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `pc_ReceivedDate` = `2026-03-20` — Claim received 278 day(s) after ec_ServiceDateFrom; payer filing window is 180 day(s) (OVER the limit; source: payer:BCBS Illinois). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `pcl_ProcedureCode` = `99214` — Procedure 99214 commonly requires modifiers ['25']; submitted modifiers: (none) - MISMATCH. (fields_referenced=['pcl_ProcedureCode', 'pcl_ProcedureModifier1', 'pcl_ProcedureModifier2', 'pcl_ProcedureModifier3', 'pcl_ProcedureModifier4'])
- `ec_PrincipalDiagnosis` = `J06.9` — Principal diagnosis J06.9 is on the supportive list (['any']) for procedure 99214; medical necessity is established. (fields_referenced=['ec_PrincipalDiagnosis'])

**Deterministic checks that fired:**

- [CRITICAL] [FAIL] `timely_filing.window_check` — Claim received 278 day(s) after ec_ServiceDateFrom; payer filing window is 180 day(s) (OVER the limit; source: payer:BCBS Illinois).
- [INFO] [PASS] `prior_auth.not_required` — No PA requirement on file for procedure 99214 with payer BCBS-IL; any CARC 197/198 denial would warrant payer-policy review.
- [CRITICAL] [FAIL] `coding.line_0.required_modifier` — Procedure 99214 commonly requires modifiers ['25']; submitted modifiers: (none) - MISMATCH.
- [INFO] [PASS] `medical_necessity.principal_supports` — Principal diagnosis J06.9 is on the supportive list (['any']) for procedure 99214; medical necessity is established.

---

## CLM-2026-00287

- **Recoverability:** `recoverable`
- **Confidence:** 0.60 (components: {'llm': 0.7, 'evidence_pass_rate': 0.5, 'blended': 0.6})
- **Model:** mock:gpt-4o | cost: $0.00000 | latency: 6 ms

**Root cause.** Denial driven by CARC 16. Procedure 27447 commonly requires modifiers ['LT', 'RT']; submitted modifiers: (none) - MISMATCH. (fields_referenced=['pcl_ProcedureCode', 'pcl_ProcedureModifier1', 'pcl_ProcedureModifier2', 'pcl_ProcedureModifier3', 'pcl_ProcedureModifier4']) Procedure 27447 is a paired-organ procedure; without LT/RT/50 modifiers, payers commonly mis-flag a legitimate second-side service as a duplicate. Resubmit with the appropriate side modifier. (fields_referenced=['pcl_ProcedureCode', 'pcl_ProcedureModifier1', 'pcl_ProcedureModifier2']) RARC N20: Service not payable with other service rendered on the same date. Missing field(s): ['pcl_ProcedureModifier1', 'pcl_ProcedureModifier2']. (fields_referenced=['pcl_ProcedureModifier1', 'pcl_ProcedureModifier2'])

**CARC interpretation.** CARC 16 (missing_info): the payer adjudicated this charge under the missing_info family of denials.

**Recommended action.** Populate the missing field(s) called out by the RARC code and resubmit a corrected claim.

**Supporting evidence (cited fields):**

- `pcl_ProcedureCode` = `27447` — Procedure 27447 commonly requires modifiers ['LT', 'RT']; submitted modifiers: (none) - MISMATCH. (fields_referenced=['pcl_ProcedureCode', 'pcl_ProcedureModifier1', 'pcl_ProcedureModifier2', 'pcl_ProcedureModifier3', 'pcl_ProcedureModifier4
- `ec_ServiceDateFrom` = `2026-01-08` — Claim received 33 day(s) after ec_ServiceDateFrom; payer filing window is 365 day(s) (WITHIN the limit; source: payer:Medicare FFS). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `pc_ReceivedDate` = `2026-02-10` — Claim received 33 day(s) after ec_ServiceDateFrom; payer filing window is 365 day(s) (WITHIN the limit; source: payer:Medicare FFS). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `ec_PriorAuthorization` = `AUTH-998877` — A prior authorization number was submitted on the claim. (fields_referenced=['ec_PriorAuthorization', 'pc_PriorAuthNum'])
- `ec_PrincipalDiagnosis` = `M17.11` — Principal diagnosis M17.11 is on the supportive list (['M17', 'Z47.1', 'M25.561', 'M25.562']) for procedure 27447; medical necessity is established. (fields_referenced=['ec_PrincipalDiagnosis'])

**Deterministic checks that fired:**

- [INFO] [PASS] `timely_filing.window_check` — Claim received 33 day(s) after ec_ServiceDateFrom; payer filing window is 365 day(s) (WITHIN the limit; source: payer:Medicare FFS).
- [INFO] [PASS] `prior_auth.present` — A prior authorization number was submitted on the claim.
- [CRITICAL] [FAIL] `coding.line_0.required_modifier` — Procedure 27447 commonly requires modifiers ['LT', 'RT']; submitted modifiers: (none) - MISMATCH.
- [INFO] [PASS] `medical_necessity.principal_supports` — Principal diagnosis M17.11 is on the supportive list (['M17', 'Z47.1', 'M25.561', 'M25.562']) for procedure 27447; medical necessity is established.
- [WARNING] [FAIL] `duplicate.line_0.bilateral_modifier_missing` — Procedure 27447 is a paired-organ procedure; without LT/RT/50 modifiers, payers commonly mis-flag a legitimate second-side service as a duplicate. Resubmit with the appropriate side modifier.
- [CRITICAL] [FAIL] `missing_info.rarc_N20` — RARC N20: Service not payable with other service rendered on the same date. Missing field(s): ['pcl_ProcedureModifier1', 'pcl_ProcedureModifier2'].

---

## CLM-2026-00391

- **Recoverability:** `recoverable`
- **Confidence:** 0.72 (components: {'llm': 0.7, 'evidence_pass_rate': 0.75, 'blended': 0.725})
- **Model:** mock:gpt-4o-mini | cost: $0.00000 | latency: 6 ms

**Root cause.** Denial driven by CARC 50. Procedure 72148 requires prior authorization for payer AETNA, but neither ec_PriorAuthorization nor pc_PriorAuthNum is populated. (fields_referenced=['ec_PriorAuthorization', 'pc_PriorAuthNum'])

**CARC interpretation.** CARC 50 (medical_necessity): the payer adjudicated this charge under the medical_necessity family of denials.

**Recommended action.** Repoint the diagnosis pointer to the supportive secondary diagnosis and resubmit; if denied again, appeal with clinical documentation.

**Supporting evidence (cited fields):**

- `ec_ServiceDateFrom` = `2026-02-20` — Claim received 9 day(s) after ec_ServiceDateFrom; payer filing window is 90 day(s) (WITHIN the limit; source: payer:Aetna Commercial). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `pc_ReceivedDate` = `2026-03-01` — Claim received 9 day(s) after ec_ServiceDateFrom; payer filing window is 90 day(s) (WITHIN the limit; source: payer:Aetna Commercial). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `ec_PrincipalDiagnosis` = `M54.5` — Principal diagnosis M54.5 is on the supportive list (['M51', 'M54.5', 'M54.4', 'M54.16', 'M54.17', 'G55', 'S33']) for procedure 72148; medical necessity is established. (fields_referenced=['ec_PrincipalDiagnosis'])

**Deterministic checks that fired:**

- [INFO] [PASS] `timely_filing.window_check` — Claim received 9 day(s) after ec_ServiceDateFrom; payer filing window is 90 day(s) (WITHIN the limit; source: payer:Aetna Commercial).
- [CRITICAL] [FAIL] `prior_auth.required_but_missing` — Procedure 72148 requires prior authorization for payer AETNA, but neither ec_PriorAuthorization nor pc_PriorAuthNum is populated.
- [INFO] [PASS] `medical_necessity.principal_supports` — Principal diagnosis M54.5 is on the supportive list (['M51', 'M54.5', 'M54.4', 'M54.16', 'M54.17', 'G55', 'S33']) for procedure 72148; medical necessity is established.
- [INFO] [PASS] `missing_info.rarc_N386` — RARC N386: This decision was based on a National Coverage Determination (NCD). All hinted fields are populated.

---

## CLM-2026-00455

- **Recoverability:** `needs_review`
- **Confidence:** 0.72 (components: {'llm': 0.7, 'evidence_pass_rate': 0.75, 'blended': 0.725})
- **Model:** mock:gpt-4o-mini | cost: $0.00000 | latency: 6 ms

**Root cause.** Denial driven by CARC 18. Procedure 99213 commonly requires modifiers ['25']; submitted modifiers: (none) - MISMATCH. (fields_referenced=['pcl_ProcedureCode', 'pcl_ProcedureModifier1', 'pcl_ProcedureModifier2', 'pcl_ProcedureModifier3', 'pcl_ProcedureModifier4'])

**CARC interpretation.** CARC 18 (duplicate): the payer adjudicated this charge under the duplicate family of denials.

**Recommended action.** Verify the line is not a true duplicate; if it represents a paired-organ or repeat service, append the appropriate modifier (LT/RT/50/59) and resubmit.

**Supporting evidence (cited fields):**

- `pcl_ProcedureCode` = `99213` — Procedure 99213 commonly requires modifiers ['25']; submitted modifiers: (none) - MISMATCH. (fields_referenced=['pcl_ProcedureCode', 'pcl_ProcedureModifier1', 'pcl_ProcedureModifier2', 'pcl_ProcedureModifier3', 'pcl_ProcedureModifier4'])
- `ec_ServiceDateFrom` = `2026-01-10` — Claim received 15 day(s) after ec_ServiceDateFrom; payer filing window is 90 day(s) (WITHIN the limit; source: payer:United Healthcare Commercial). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `pc_ReceivedDate` = `2026-01-25` — Claim received 15 day(s) after ec_ServiceDateFrom; payer filing window is 90 day(s) (WITHIN the limit; source: payer:United Healthcare Commercial). (fields_referenced=['ec_ServiceDateFrom', 'pc_ReceivedDate'])
- `ec_PrincipalDiagnosis` = `J20.9` — Principal diagnosis J20.9 is on the supportive list (['any']) for procedure 99213; medical necessity is established. (fields_referenced=['ec_PrincipalDiagnosis'])

**Deterministic checks that fired:**

- [INFO] [PASS] `timely_filing.window_check` — Claim received 15 day(s) after ec_ServiceDateFrom; payer filing window is 90 day(s) (WITHIN the limit; source: payer:United Healthcare Commercial).
- [INFO] [PASS] `prior_auth.not_required` — No PA requirement on file for procedure 99213 with payer UHC; any CARC 197/198 denial would warrant payer-policy review.
- [CRITICAL] [FAIL] `coding.line_0.required_modifier` — Procedure 99213 commonly requires modifiers ['25']; submitted modifiers: (none) - MISMATCH.
- [INFO] [PASS] `medical_necessity.principal_supports` — Principal diagnosis J20.9 is on the supportive list (['any']) for procedure 99213; medical necessity is established.

---
