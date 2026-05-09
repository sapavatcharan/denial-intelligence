"""Pure metrics used by the eval harness.

These have no project-specific deps so they're easy to unit-test.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    out: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        out[t][p] += 1
    return out


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
        f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return round(macro, 3), per_class


def brier_score(y_true_one_hot: list[float], y_pred_prob: list[float]) -> float:
    """Mean squared error between predicted probability and outcome (0/1)."""
    if not y_true_one_hot:
        return 0.0
    return round(
        sum((p - t) ** 2 for p, t in zip(y_pred_prob, y_true_one_hot, strict=True))
        / len(y_true_one_hot),
        4,
    )


def evidence_grounding_rate(citations_per_claim: list[list[bool]]) -> float:
    """Fraction of cited fields that pass the grounding gate."""
    flat = [b for cl in citations_per_claim for b in cl]
    if not flat:
        return 1.0
    return round(sum(flat) / len(flat), 3)


def per_scenario_breakdown(scenarios: list[str], correct_flags: list[bool]) -> dict[str, dict[str, float]]:
    """Per-scenario counts and accuracy."""
    by_scenario: dict[str, list[bool]] = defaultdict(list)
    for s, ok in zip(scenarios, correct_flags, strict=True):
        by_scenario[s].append(ok)
    out: dict[str, dict[str, float]] = {}
    for s, flags in by_scenario.items():
        out[s] = {
            "n": float(len(flags)),
            "correct": float(sum(flags)),
            "accuracy": round(sum(flags) / len(flags), 3),
        }
    return out


def label_distribution(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))
