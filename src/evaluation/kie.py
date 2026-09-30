"""Field-level evaluation for KIE predictions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from src.extraction.normalization import values_match
from src.extraction.types import ExtractionResult

FIELDS = ("company", "date", "address", "total")


def evaluate_extractions(
    predictions: Iterable[ExtractionResult],
    ground_truth: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """Return exact, normalized, precision, recall, and F1 per field."""

    prediction_list = list(predictions)
    truth_list = list(ground_truth)
    if len(prediction_list) != len(truth_list):
        raise ValueError("predictions and ground_truth must have the same length")

    rows = []
    for field in FIELDS:
        exact = normalized = true_positive = false_positive = false_negative = 0
        for prediction, truth in zip(prediction_list, truth_list):
            expected = str(truth.get(field, "") or "")
            extracted = prediction.fields()[field].value or ""
            if expected == extracted and expected:
                exact += 1
            matches = values_match(expected, extracted, monetary=field == "total")
            if matches:
                normalized += 1
                true_positive += 1
            elif extracted:
                false_positive += 1
                if expected:
                    false_negative += 1
            elif expected:
                false_negative += 1
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "field": field,
                "exact_match": exact / len(truth_list) if truth_list else 0.0,
                "normalized_exact_match": normalized / len(truth_list) if truth_list else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    metric_rows = pd.DataFrame(rows)
    macro_row = {
        "field": "macro_avg",
        "exact_match": metric_rows["exact_match"].mean() if rows else 0.0,
        "normalized_exact_match": metric_rows["normalized_exact_match"].mean() if rows else 0.0,
        "precision": metric_rows["precision"].mean() if rows else 0.0,
        "recall": metric_rows["recall"].mean() if rows else 0.0,
        "f1": metric_rows["f1"].mean() if rows else 0.0,
    }
    rows.append(macro_row)
    return pd.DataFrame(rows).set_index("field")