import pandas as pd

from src.utils import get_categorical_indices, unravel_metric_report


def test_get_categorical_indices():
    data = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            "C": [1.1, 2.2, 3.3],
            "D": [True, False, True],
            "E": ["x", "y", "z"],
        }
    )

    metadata = {
        "fields": {
            "A": {"type": "numeric"},
            "B": {"type": "categorical"},
            "C": {"type": "numeric"},
            "D": {"type": "boolean"},
            "E": {"type": "categorical"},
        }
    }

    expected_indices = [1, 3, 4]
    actual_indices = get_categorical_indices(data, metadata)

    assert expected_indices == actual_indices


def test_unravel_metric_report():
    report_dict = {
        "0": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 150},
        "1": {"precision": 0.75, "recall": 0.5, "f1-score": 0.6, "support": 81},
        "accuracy": 0.77,
        "macro avg": {
            "precision": 0.76,
            "recall": 0.71,
            "f1-score": 0.73,
            "support": 231,
        },
        "weighted avg": {
            "precision": 0.76,
            "recall": 0.77,
            "f1-score": 0.75,
            "support": 231,
        },
    }

    expected_output = {
        "0_precision": 0.8,
        "0_recall": 0.9,
        "0_f1-score": 0.85,
        "0_support": 150,
        "1_precision": 0.75,
        "1_recall": 0.5,
        "1_f1-score": 0.6,
        "1_support": 81,
        "accuracy": 0.77,
        "macro avg_precision": 0.76,
        "macro avg_recall": 0.71,
        "macro avg_f1-score": 0.73,
        "macro avg_support": 231,
        "weighted avg_precision": 0.76,
        "weighted avg_recall": 0.77,
        "weighted avg_f1-score": 0.75,
        "weighted avg_support": 231,
    }

    result = unravel_metric_report(report_dict)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
