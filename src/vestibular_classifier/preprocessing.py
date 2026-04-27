"""Data loading, label canonicalisation, and cohort filtering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _derive_age_years(df: pd.DataFrame) -> pd.Series:
    """Return a Series of ages in years computed from ``D_Age`` and ``D_Time_final``."""
    def _one(row: pd.Series) -> int:
        s = str(int(row["D_Age"]))
        if len(s) < 6:
            s = "0" + s
        yy = int(s[:2])
        birth_year = 2000 + yy if yy < 20 else 1900 + yy
        return int(row["D_Time_final"].year - birth_year)

    return df.apply(_one, axis=1).astype(int)


def load_responses(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw response CSV and derive age in years."""
    df = pd.read_csv(csv_path)
    df["D_Time_final"] = pd.to_datetime(df["D_Time_final"])
    df["D_Age_years"] = _derive_age_years(df)
    return df


def filter_multiclass_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the multi-class cohort exclusions described in Materials and Methods.

    Removes:
      * subjects below the study's minimum age (``D_Age_years < 20``)
      * subjects whose primary reference diagnosis is ``OTHERS`` (out-of-scope)
    """
    out = df[df["Ref_Dx1"] != "OTHERS"]
    out = out[out["D_Age_years"] >= 20]
    return out.reset_index(drop=True)


def filter_vascular_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the vascular-branch cohort exclusion (age filter only).
    """
    out = df[df["D_Age_years"] >= 20]
    return out.reset_index(drop=True)
