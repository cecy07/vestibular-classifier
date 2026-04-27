"""Accept/reject post-processing rules applied to the ranked scoring output.

These rules refine the initial top-two candidates produced by
:mod:`vestibular_classifier.scoring` with the Rule-based algorithm section and the Discussion of the paper.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _record(df_audit: pd.DataFrame, column: str, idx, label: str) -> None:
    """Append ``label`` to ``df_audit[column][idx]``, space-separated."""
    current = df_audit.at[idx, column]
    df_audit.at[idx, column] = f"{current} {label}".strip()


def _insert_dx(df: pd.DataFrame, idx, dx: str, *, at_top: bool = False) -> None:
    """Insert ``dx`` into App_Dx1/App_Dx2 if not already present."""
    d1 = df.at[idx, "App_Dx1"]
    d2 = df.at[idx, "App_Dx2"]
    if d1 == dx or d2 == dx:
        return
    if d1 == "":
        df.at[idx, "App_Dx1"] = dx
    elif d2 == "":
        df.at[idx, "App_Dx2"] = dx
    elif at_top:
        df.at[idx, "App_Dx2"] = d1
        df.at[idx, "App_Dx1"] = dx
    else:
        df.at[idx, "App_Dx2"] = dx


def _clear_dx(df: pd.DataFrame, idx, dx: str) -> None:
    """Remove ``dx`` from either Dx slot if it is present there."""
    if df.at[idx, "App_Dx1"] == dx:
        df.at[idx, "App_Dx1"] = ""
    if df.at[idx, "App_Dx2"] == dx:
        df.at[idx, "App_Dx2"] = ""


# --------------------------------------------------------------------------
# Rejection rules
# --------------------------------------------------------------------------

def PPPD_reject_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """>= 4 "No" answers among Q14-Q20 -> reject PPPD."""
    qs = ["Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20"]
    for idx, row in df.iterrows():
        if sum(row[q] == 2 for q in qs) >= 4:
            _record(audit, "Reject", idx, "PPPD")
            _clear_dx(df, idx, "PPPD")


def MD_reject_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Q03!=2 AND Q04!=1 AND Q20-Q22 all "No" -> reject MD."""
    for idx, row in df.iterrows():
        if row["Q03"] != 2 and row["Q04"] != 1 \
                and row["Q20"] != 1 and row["Q21"] != 1 and row["Q22"] != 1:
            _record(audit, "Reject", idx, "MD")
            _clear_dx(df, idx, "MD")


def MD_reject_2(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Secondary MD with only one "Yes" among Q20-Q22 -> reject."""
    for idx, row in df.iterrows():
        if row["App_Dx2"] != "MD":
            continue
        q20, q21, q22 = row["Q20"], row["Q21"], row["Q22"]
        single_yes = (
            (q20 == 1 and q21 == 2 and q22 == 2)
            or (q20 == 2 and q21 == 1 and q22 == 2)
            or (q20 == 2 and q21 == 2 and q22 == 1)
        )
        if single_yes:
            _record(audit, "Reject", idx, "MD")
            df.at[idx, "App_Dx2"] = ""


def VM_reject_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """App_Dx1==OH AND App_Dx2==VM -> reject VM (OH takes precedence)."""
    for idx, row in df.iterrows():
        if row["App_Dx1"] == "OH" and row["App_Dx2"] == "VM":
            _record(audit, "Reject", idx, "VM")
            df.at[idx, "App_Dx2"] = ""


def VM_reject_3(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """< 3 "Yes" across the migraine-symptom cluster -> reject VM."""
    qs = ["Q25", "Q27", "Q28", "Q29", "Q32", "Q33", "Q35", "Q36", "Q37", "Q38"]
    for idx, row in df.iterrows():
        if sum(row[q] == 1 for q in qs) < 3:
            _record(audit, "Reject", idx, "VM")
            _clear_dx(df, idx, "VM")


def VEST_reject_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Q14 and Q15 both "No" -> reject VEST."""
    for idx, row in df.iterrows():
        if row["Q14"] == 2 and row["Q15"] == 2:
            _record(audit, "Reject", idx, "VEST")
            _clear_dx(df, idx, "VEST")


def OH_reject_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Q45, Q47, Q48 all "Yes" -> reject OH.

    Paired with :func:`OH_accept_3` below: together they act as a more
    specific OH gate, firing only when the additional Q04==1 and Q05!=3
    context is also present.
    """
    for idx, row in df.iterrows():
        if row["Q45"] == 1 and row["Q47"] == 1 and row["Q48"] == 1:
            _record(audit, "Reject", idx, "OH")
            _clear_dx(df, idx, "OH")


# --------------------------------------------------------------------------
# Acceptance rules
# --------------------------------------------------------------------------

def VM_accept_2(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """>= 3 "Yes" among Q35, Q37, Q38, Q28, Q32 -> accept VM."""
    qs = ["Q35", "Q37", "Q38", "Q28", "Q32"]
    for idx, row in df.iterrows():
        if sum(row[q] == 1 for q in qs) >= 3:
            _record(audit, "Accept", idx, "VM")
            _insert_dx(df, idx, "VM")


def VEST_accept_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """(Q05==1 OR Q05==2) AND Q04!=3 -> accept VEST."""
    for idx, row in df.iterrows():
        if (row["Q05"] == 1 or row["Q05"] == 2) and row["Q04"] != 3:
            _record(audit, "Accept", idx, "VEST")
            _insert_dx(df, idx, "VEST")


def PPPD_accept_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Q02==2 AND Q15-Q18 all "Yes" -> accept PPPD."""
    for idx, row in df.iterrows():
        if (row["Q02"] == 2 and row["Q15"] == 1 and row["Q16"] == 1
                and row["Q17"] == 1 and row["Q18"] == 1):
            _record(audit, "Accept", idx, "PPPD")
            _insert_dx(df, idx, "PPPD")


def OH_accept_3(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Q45,Q47,Q48 "Yes" AND Q05!=3 AND Q04==1 -> accept OH."""
    for idx, row in df.iterrows():
        if (row["Q45"] == 1 and row["Q47"] == 1 and row["Q48"] == 1
                and row["Q05"] != 3 and row["Q04"] == 1):
            _record(audit, "Accept", idx, "OH")
            _insert_dx(df, idx, "OH")


def BPPV_accept_1(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    """>= 2 "Yes" among {Q05==1, Q23, Q24} -> accept BPPV (promoted to top)."""
    for idx, row in df.iterrows():
        count = int(row["Q05"] == 1) + int(row["Q23"] == 1) + int(row["Q24"] == 1)
        if count >= 2:
            _record(audit, "Accept", idx, "BPPV")
            _insert_dx(df, idx, "BPPV", at_top=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

_REJECTIONS = {
    "PPPD_reject_1": PPPD_reject_1,
    "MD_reject_1":   MD_reject_1,
    "MD_reject_2":   MD_reject_2,
    "VM_reject_1":   VM_reject_1,
    "VM_reject_3":   VM_reject_3,
    "VEST_reject_1": VEST_reject_1,
    "OH_reject_1":   OH_reject_1,
}

_ACCEPTANCES = {
    "VM_accept_2":   VM_accept_2,
    "VEST_accept_1": VEST_accept_1,
    "PPPD_accept_1": PPPD_accept_1,
    "OH_accept_3":   OH_accept_3,
    "BPPV_accept_1": BPPV_accept_1,
}


def apply_rules(
    df: pd.DataFrame,
    rejections: Mapping[str, bool] | None = None,
    acceptances: Mapping[str, bool] | None = None,
    promote_empty_dx1: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply rejection and acceptance rules in the canonical order.

    The input must contain the columns ``App_Dx1`` and ``App_Dx2`` (produced
    by renaming ``R_Dx1`` / ``R_Dx2`` from the scoring stage) plus the
    question columns ``Q01`` ... ``Q50``.

    Returns
    -------
    (updated_df, audit_df)
        ``updated_df`` is a copy of ``df`` with modified Dx columns and
        two new columns ``Accept`` / ``Reject`` summarising which rules
        fired for each subject.
    """
    rejections = dict(rejections) if rejections else {}
    acceptances = dict(acceptances) if acceptances else {}

    work = df.copy()
    work["Accept"] = ""
    work["Reject"] = ""
    audit = work[["Accept", "Reject"]].copy()

    for name, fn in _REJECTIONS.items():
        if rejections.get(name, True):
            fn(work, audit)

    if promote_empty_dx1:
        empty = work["App_Dx1"] == ""
        work.loc[empty, "App_Dx1"] = work.loc[empty, "App_Dx2"]
        work.loc[empty, "App_Dx2"] = ""

    for name, fn in _ACCEPTANCES.items():
        if acceptances.get(name, True):
            fn(work, audit)

    work["Accept"] = audit["Accept"]
    work["Reject"] = audit["Reject"]
    return work, audit
