"""Rule-based classifier for six ICVD-defined vestibular disorders.

Companion code for the paper:
"An Advanced Rule-Based Mobile Classifier for the Automated Diagnosis of
Vestibular Disorders: A Pilot Study" (Ryu, Callejas Pastor, Joo, Ku, Suh).

Typical usage::

    from vestibular_classifier import (
        load_responses, filter_multiclass_cohort,
        score_dataframe, apply_rules,
        score_top_two, per_class_metrics,
        evaluate_vascular,
        load_coefficients, load_postprocessing,
    )

    coeffs = load_coefficients()
    post = load_postprocessing()

    raw = load_responses("data/responses.csv")
    cohort = filter_multiclass_cohort(raw)

    scored = score_dataframe(cohort, coeffs)
    scored = scored.rename(columns={"R_Dx1": "App_Dx1", "R_Dx2": "App_Dx2"})
    final, _audit = apply_rules(scored, post["rejections"], post["acceptances"])

    print(score_top_two(final).as_dict())
"""

from .config import load_coefficients, load_postprocessing
from .evaluation import (
    PerClassMetrics,
    TopTwoResult,
    format_per_class_report,
    format_top_two_report,
    per_class_metrics,
    score_top_two,
)
from .postprocessing import apply_rules
from .preprocessing import (
    filter_multiclass_cohort,
    filter_vascular_cohort,
    load_responses,
)
from .scoring import DX_CLASSES, OTHERS_LABEL, score_dataframe
from .vascular import VascularMetrics, evaluate_vascular, predict_vascular

__all__ = [
    "DX_CLASSES",
    "OTHERS_LABEL",
    "PerClassMetrics",
    "TopTwoResult",
    "VascularMetrics",
    "apply_rules",
    "evaluate_vascular",
    "filter_multiclass_cohort",
    "filter_vascular_cohort",
    "format_per_class_report",
    "format_top_two_report",
    "load_coefficients",
    "load_postprocessing",
    "load_responses",
    "per_class_metrics",
    "predict_vascular",
    "score_dataframe",
    "score_top_two",
]

__version__ = "1.0.0"
