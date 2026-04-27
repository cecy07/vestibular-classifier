"""Rule-based scoring for the six target ICVD-defined vestibular disorders.

Implements equations of the manuscript: each class starts with a
base weight of 1.0 and is multiplied by a class-specific coefficient whenever
a question (or question set) satisfies the corresponding rule.

Classes
-------
BPPV : benign paroxysmal positional vertigo
MD   : Meniere's disease
VM   : vestibular migraine
PPPD : persistent postural-perceptual dizziness
VEST : peripheral vestibulopathy (composite of AUVP / vestibular neuritis and
       bilateral vestibulopathy)
OH   : orthostatic dizziness

``OTHERS`` is used as the out-of-scope label for subjects whose reference
diagnosis falls outside the six target classes; it is not produced by
the classifier.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


# The six target classes, in the order used by the ``scoreA_``/``scoreB_``
# output columns.
DX_CLASSES: tuple[str, ...] = ("BPPV", "MD", "VM", "PPPD", "VEST", "OH")

# Label for the out-of-scope reference diagnosis.
OTHERS_LABEL: str = "OTHERS"


def _compute_age_years(d_age: int | float, year_of_visit: int) -> int:
    """Recover age in years from the 6-digit Korean-style birth-date field.

    ``D_Age`` is encoded as YYMMDD where the century is inferred from YY
    (``YY < 20`` -> 2000s, otherwise 1900s).
    """
    s = str(int(d_age))
    if len(s) < 6:
        s = "0" + s
    yy = int(s[:2])
    birth_year = 2000 + yy if yy < 20 else 1900 + yy
    return int(year_of_visit - birth_year)


def _vascular_risk_score(row: pd.Series, is_old: bool) -> int:
    """Vascular risk count used internally (different from the top-level
    screen in ``vascular.py``; this one mirrors the per-subject book-keeping
    that the original algorithm recorded in ``App_vasScore``)."""
    score = 0
    for q in ("Q06", "Q07", "Q08", "Q09", "Q10", "Q13"):
        if row[q] == 1:
            score += 1
    q11_yes = row["Q11"] == 1
    q12_yes = row["Q12"] == 1
    if q11_yes and q12_yes:
        score += 1
    elif q11_yes or q12_yes:
        if is_old:
            score += 1
    return score


def _score_one_subject(row: pd.Series, v: Mapping[str, float]) -> dict:
    """Apply the full rule set to one subject's question responses."""
    # Base weights (eq. 2 starts each class at 1).
    w_bppv = 1.0
    w_md = 1.0
    w_vm = 1.0
    w_pppd = 1.0
    w_vest = 1.0
    w_oh = 1.0

    age_years = _compute_age_years(row["D_Age"], row["D_Time_final"].year)
    is_old = age_years >= 65
    vasc_score = _vascular_risk_score(row, is_old)
    is_vascular_risk = vasc_score >= 4

    # --- Q03 ---------------------------------------------------------------
    if row["Q03"] == 1:
        w_bppv *= v["xBPPV_1"]

    # --- Q04 ---------------------------------------------------------------
    if row["Q04"] == 1:
        w_bppv *= v["xBPPV_2"]
        w_md *= v["xMD_1"]
        w_vm *= v["xVM_1"]
        w_vest *= v["xVEST_1"]
    elif row["Q04"] == 2:
        w_md *= v["xMD_2"]
        w_vm *= v["xVM_2"]
        w_vest *= v["xVEST_2"]
    elif row["Q04"] == 3:
        w_md *= v["xMD_3"]
        w_vm *= v["xVM_3"]
        w_pppd *= v["xPPPD_1"]
        w_vest *= v["xVEST_3"]

    # --- Q05 ---------------------------------------------------------------
    if row["Q05"] == 1:
        w_bppv *= v["xBPPV_3"]
        w_md *= v["xMD_4"]
        w_vm *= v["xVM_4"]
        w_pppd *= v["xPPPD_2"]
        w_vest *= v["xVEST_4"]
    elif row["Q05"] == 2:
        w_pppd *= v["xPPPD_3"]
        w_vest *= v["xVEST_5"]
    elif row["Q05"] == 3:
        w_pppd *= v["xPPPD_4"]
        w_vest *= v["xVEST_6"]
        w_oh *= v["xOH_1"]

    # --- Question-set rules ------------------------------------------------
    # PPPD: Q14-Q19 "Yes" count.
    n_pppd = sum(1 for i in range(14, 20) if row[f"Q{i:02d}"] == 1)
    w_pppd *= v["xPPPD_5"] * n_pppd

    # MD: Q20-Q22 "Yes" count.
    n_md = sum(1 for i in range(20, 23) if row[f"Q{i:02d}"] == 1)
    w_md *= v["xMD_5"] * n_md

    # BPPV: Q23-Q24 "Yes" count.
    n_bppv = sum(1 for i in range(23, 25) if row[f"Q{i:02d}"] == 1)
    w_bppv *= v["xBPPV_4"] * n_bppv

    # VM: Q25 gates the headache block (Q26-Q29, Q31-Q39).
    if row["Q25"] == 2:
        w_vm /= v["xVM_5"]
    elif row["Q25"] == 1:
        n_vm = (
            sum(1 for i in range(26, 30) if row[f"Q{i:02d}"] == 1)
            + sum(1 for i in range(31, 40) if row[f"Q{i:02d}"] == 1)
        )
        w_vm *= v["xVM_6"] * n_vm

    # VEST: Q41-Q44 "Yes" count.
    n_vest = sum(1 for i in range(41, 45) if row[f"Q{i:02d}"] == 1)
    w_vest *= v["xVEST_7"] * n_vest

    # OH: Q45-Q50 "Yes" count.
    n_oh = sum(1 for i in range(45, 51) if row[f"Q{i:02d}"] == 1)
    w_oh *= v["xOH_2"] * n_oh

    # --- Normalise to percentages (eq. 3) ----------------------------------
    raw = np.array([w_bppv, w_md, w_vm, w_pppd, w_vest, w_oh], dtype=float)
    total = raw.sum()
    if total <= 0:
        pct = np.zeros_like(raw)
    else:
        pct = np.round(raw / total * 100, 0)

    order = np.argsort(-pct)
    ranked_names = [DX_CLASSES[i] for i in order]
    ranked_scores = pct[order]

    return {
        "R_Dx1": ranked_names[0],
        "R_Dx2": ranked_names[1],
        "R_Dx3": ranked_names[2],
        "R_Dx1Score": float(ranked_scores[0]),
        "R_Dx2Score": float(ranked_scores[1]),
        "R_Dx3Score": float(ranked_scores[2]),
        "App_isOld": int(is_old),
        "App_isVas": int(is_vascular_risk),
        "App_vasScore": int(vasc_score),
        "scoreA_BPPV": float(w_bppv),
        "scoreB_MD": float(w_md),
        "scoreC_VM": float(w_vm),
        "scoreD_PPPD": float(w_pppd),
        "scoreE_VEST": float(w_vest),
        "scoreF_OH": float(w_oh),
    }


def score_dataframe(df: pd.DataFrame, coefficients: Mapping[str, float]) -> pd.DataFrame:
    """Score every row of ``df`` and return a copy with the new columns attached."""
    out = df.copy()
    records = [_score_one_subject(row, coefficients) for _, row in out.iterrows()]
    scored = pd.DataFrame.from_records(records, index=out.index)
    return pd.concat([out, scored], axis=1)
