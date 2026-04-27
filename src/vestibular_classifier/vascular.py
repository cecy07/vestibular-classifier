"""Vascular-vertigo safety screen

The screen fires when the number of "Yes" answers across a fixed set of
risk questions meets or exceeds a threshold.

This module implements the screen as a function of (questions, threshold)
and provides a small evaluator for computing accuracy, sensitivity and
specificity against the expert ``Ref_Vas`` ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class VascularMetrics:
    """Container for vascular-branch performance metrics."""
    n: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    sensitivity: float
    specificity: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


def predict_vascular(
    df: pd.DataFrame,
    questions: Sequence[str] = ("Q06", "Q07", "Q08", "Q13"),
    threshold: int = 2,
) -> pd.Series:
    """Return a boolean Series indicating the screen outcome per subject.

    True means the screen flags the subject as at-risk for vascular vertigo.
    The count is over "Yes" answers (coded 1) in ``questions``; the screen
    fires when ``count >= threshold``.
    """
    missing = [q for q in questions if q not in df.columns]
    if missing:
        raise KeyError(f"Required question columns missing: {missing}")
    yes_count = (df[list(questions)] == 1).sum(axis=1)
    return yes_count >= threshold


def evaluate_vascular(
    df: pd.DataFrame,
    questions: Sequence[str] = ("Q06", "Q07", "Q08", "Q13"),
    threshold: int = 2,
    truth_col: str = "Ref_Vas",
) -> VascularMetrics:
    """Run the screen and compare against the expert ground-truth column.

    ``truth_col`` can be either a boolean column or one coded as "Yes"/"No".
    Returns a :class:`VascularMetrics` dataclass; use ``.as_dict()`` for
    serialisation.
    """
    pred = predict_vascular(df, questions=questions, threshold=threshold)
    truth = df[truth_col]
    if truth.dtype == object:
        truth = truth.astype(str).str.lower().map({"yes": True, "no": False})
    truth = truth.astype(bool)

    tp = int(((pred) & (truth)).sum())
    tn = int(((~pred) & (~truth)).sum())
    fp = int(((pred) & (~truth)).sum())
    fn = int(((~pred) & (truth)).sum())

    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else float("nan")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    return VascularMetrics(
        n=n, tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=acc, sensitivity=sens, specificity=spec,
    )
