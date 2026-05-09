# Synthetic Claim Generator - Optional LLM Helper Prompt

The synthetic dataset is built primarily by rule-based templates in `scripts/generate_synthetic.py`. This prompt is used only to vary the *naturalistic* free-text fields so that two generated claims do not look identical to a reviewer.

## Rules

- All generated PHI is fake. Names should be obviously synthetic (e.g., `John Sample`, `Jane Test`).
- Provider NPIs must be 10 digits and not match any real provider. Prefix with `99999` to make this obvious.
- Do not invent diagnosis or procedure codes. Use only the codes the caller provides.
- Do not change dates the caller provides - only fill in missing optional fields.

## Output

A JSON object with only the fields the caller asked you to fill, no extra keys.
