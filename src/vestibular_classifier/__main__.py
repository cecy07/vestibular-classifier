"""Command-line entry point for reproducing the paper's headline results.

Run from the repository root after installing the package::

    python -m vestibular_classifier --data data/responses.csv

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import (
    apply_rules,
    evaluate_vascular,
    filter_multiclass_cohort,
    filter_vascular_cohort,
    format_per_class_report,
    format_top_two_report,
    load_coefficients,
    load_postprocessing,
    load_responses,
    per_class_metrics,
    score_dataframe,
    score_top_two,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce the paper's classification results from a responses CSV."
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the responses CSV (schema documented in data/README.md).",
    )
    p.add_argument(
        "--coefficients",
        type=Path,
        default=None,
        help="Override path to coefficients.yaml (default: configs/coefficients.yaml).",
    )
    p.add_argument(
        "--postprocessing",
        type=Path,
        default=None,
        help="Override path to postprocessing.yaml (default: configs/postprocessing.yaml).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write per-subject classifier output and summary CSVs.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coeffs = load_coefficients(args.coefficients)
    post = load_postprocessing(args.postprocessing)

    raw = load_responses(args.data)
    print(f"Loaded {len(raw)} subjects from {args.data}.\n")

    # --- Multi-class analysis ---------------------------------------------
    multi = filter_multiclass_cohort(raw)
    print(f"Multi-class cohort after exclusions: n = {len(multi)}")

    scored = score_dataframe(multi, coeffs)
    scored = scored.rename(columns={"R_Dx1": "App_Dx1", "R_Dx2": "App_Dx2"})
    final, _audit = apply_rules(
        scored,
        rejections=post.get("rejections"),
        acceptances=post.get("acceptances"),
        promote_empty_dx1=post.get("promote_empty_dx1", True),
    )

    top_two = score_top_two(final)
    print("\n--- Top-two agreement ---")
    print(format_top_two_report(top_two))

    diagnoses = ["BPPV", "VM", "VEST", "MD", "PPPD", "OH"]
    per_class = per_class_metrics(final, diagnoses)
    print("\n--- Per-class binary metrics ---")
    print(format_per_class_report(per_class))

    # --- Vascular branch --------------------------------------------------
    vasc_cohort = filter_vascular_cohort(raw)
    vs = post.get("vascular_screen", {})
    vm = evaluate_vascular(
        vasc_cohort,
        questions=vs.get("questions", ["Q06", "Q07", "Q08", "Q13"]),
        threshold=vs.get("threshold", 2),
    )
    print(f"\n--- Vascular screen (n = {vm.n}) ---")
    print(
        f"Accuracy:    {vm.accuracy:.3f}\n"
        f"Sensitivity: {vm.sensitivity:.3f}\n"
        f"Specificity: {vm.specificity:.3f}\n"
        f"TP={vm.tp}  TN={vm.tn}  FP={vm.fp}  FN={vm.fn}"
    )

    # --- Write CSVs -------------------------------------------------------
    final.to_csv(args.output_dir / "predictions_multiclass.csv", index=False)
    pd.DataFrame([top_two.as_dict()]).to_csv(
        args.output_dir / "summary_top_two.csv", index=False
    )
    pd.DataFrame([m.as_dict() for m in per_class.values()]).to_csv(
        args.output_dir / "summary_per_class.csv", index=False
    )
    pd.DataFrame([vm.as_dict()]).to_csv(
        args.output_dir / "summary_vascular.csv", index=False
    )
    print(f"\nResults written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
