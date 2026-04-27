"""Evaluation helpers: top-two scoring and per-class confusion matrices.

    Correct   if App_Dx1 matches the reference primary diagnosis
    Partial   if App_Dx1 matches the reference secondary diagnosis, or if
              App_Dx2 matches either reference diagnosis
    Incorrect otherwise
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


@dataclass
class TopTwoResult:
    """Counts and proportions for the top-two agreement outcome."""
    correct: int
    partial: int
    incorrect: int

    @property
    def total(self) -> int:
        return self.correct + self.partial + self.incorrect

    @property
    def accuracy(self) -> float:
        """Fraction of cases that are Correct or Partial (as reported in the paper)."""
        return (self.correct + self.partial) / self.total if self.total else float("nan")

    def as_dict(self) -> dict:
        return {
            "correct": self.correct,
            "partial": self.partial,
            "incorrect": self.incorrect,
            "total": self.total,
            "correct_pct": self.correct / self.total * 100 if self.total else float("nan"),
            "partial_pct": self.partial / self.total * 100 if self.total else float("nan"),
            "incorrect_pct": self.incorrect / self.total * 100 if self.total else float("nan"),
            "top_two_accuracy_pct": self.accuracy * 100,
        }


def score_top_two(
    df: pd.DataFrame,
    Ref_dx1: str = "Ref_Dx1",
    Ref_dx2: str = "Ref_Dx2",
    app_dx1: str = "App_Dx1",
    app_dx2: str = "App_Dx2",
) -> TopTwoResult:
    """Classify each row as Correct / Partial / Incorrect.

    The checks are applied in the order: Partial-from-Dx2, Partial-from-Dx1
    cross-match, then Correct. This matches the
    behaviour of the original ``check_res`` helper.
    """
    correct = partial = incorrect = 0
    for _, row in df.iterrows():
        verdict = "Incorrect"
        d1, d2 = row[Ref_dx1], row[Ref_dx2]
        a1, a2 = row[app_dx1], row[app_dx2]

        if d2 == a1 or d2 == a2:
            verdict = "Partial"
        if d1 == a2:
            verdict = "Partial"
        if d1 == a1:
            verdict = "Correct"

        if verdict == "Correct":
            correct += 1
        elif verdict == "Partial":
            partial += 1
        else:
            incorrect += 1

    return TopTwoResult(correct=correct, partial=partial, incorrect=incorrect)


@dataclass
class PerClassMetrics:
    """Binary classification metrics for a single disorder class."""
    diagnosis: str
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    sensitivity: float
    specificity: float

    def as_dict(self) -> dict:
        return {
            "diagnosis": self.diagnosis,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


def _binary_labels(df: pd.DataFrame, cols: Iterable[str], target: str) -> np.ndarray:
    """1 if any of the listed columns equals ``target`` for that row, else 0."""
    cols = list(cols)
    mask = np.zeros(len(df), dtype=int)
    for c in cols:
        mask |= (df[c].astype(str) == target).to_numpy().astype(int)
    return mask


def per_class_metrics(
    df: pd.DataFrame,
    diagnoses: Iterable[str],
    ref_cols: Iterable[str] = ("Ref_Dx1", "Ref_Dx2"),
    pred_cols: Iterable[str] = ("App_Dx1", "App_Dx2"),
) -> dict[str, PerClassMetrics]:
    """Compute binary confusion-matrix metrics per disorder class.

    For each diagnosis a subject is marked "positive" if the class appears
    in either of the reference columns (or predicted columns). This mirrors
    Figure 3 of the paper.
    """
    ref_cols = list(ref_cols)
    pred_cols = list(pred_cols)
    out: dict[str, PerClassMetrics] = {}

    for dx in diagnoses:
        y_true = _binary_labels(df, ref_cols, dx)
        y_pred = _binary_labels(df, pred_cols, dx)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")

        out[dx] = PerClassMetrics(
            diagnosis=dx, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            accuracy=float(acc), sensitivity=float(sens), specificity=float(spec),
        )

    return out


def format_top_two_report(result: TopTwoResult) -> str:
    """Plain-text summary matching the style of the paper's Results section."""
    d = result.as_dict()
    return (
        f"Total cases:  {d['total']}\n"
        f"Correct:      {d['correct']:>3}  ({d['correct_pct']:.1f}%)\n"
        f"Partial:      {d['partial']:>3}  ({d['partial_pct']:.1f}%)\n"
        f"Incorrect:    {d['incorrect']:>3}  ({d['incorrect_pct']:.1f}%)\n"
        f"Top-2 acc.:   {d['top_two_accuracy_pct']:.1f}%"
    )


def format_per_class_report(metrics: Mapping[str, PerClassMetrics]) -> str:
    """Plain-text per-disorder table matching Figure 3 of the paper."""
    lines = [
        f"{'Class':<6} {'Acc':>5} {'Sens':>5} {'Spec':>5} "
        f"{'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}"
    ]
    for dx, m in metrics.items():
        lines.append(
            f"{dx:<6} {m.accuracy:>5.2f} {m.sensitivity:>5.2f} {m.specificity:>5.2f} "
            f"{m.tn:>4} {m.fp:>4} {m.fn:>4} {m.tp:>4}"
        )
    return "\n".join(lines)
