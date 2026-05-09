"""Evaluation harness: run pipeline against the gold set and report metrics."""

from .harness import EvalReport, EvalRow, run_eval
from .metrics import (
    accuracy,
    brier_score,
    confusion_matrix,
    evidence_grounding_rate,
    macro_f1,
)

__all__ = [
    "EvalReport",
    "EvalRow",
    "accuracy",
    "brier_score",
    "confusion_matrix",
    "evidence_grounding_rate",
    "macro_f1",
    "run_eval",
]
