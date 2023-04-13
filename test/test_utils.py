import pandas as pd
import numpy as np
import pytest

from src.utils import (get_categorical_indices, 
                       unravel_metric_report, 
                       extract_loss_info_from_stdout,
                       convert_and_clean_dict)


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

@pytest.fixture
def input_str():
    return """#START#
SD0Q1_0
Epoch 1, Loss G: -0.0437,Loss D: -0.5837
Epoch 2, Loss G: -0.1434,Loss D:  0.3780
#END#
#START#
SD0Q1_1
Epoch 1, Loss G: -0.0437,Loss D: -0.5837
Epoch 2, Loss G: -0.1434,Loss D:  0.3780
#END#
#START#
SD0Q2_0
Epoch 1, Loss G: -0.0437,Loss D: -0.5837
Epoch 2, Loss G: -0.1434,Loss D:  0.3780
Epoch 3, Loss G: -0.1649,Loss D:  0.0226
Epoch 4, Loss G: -0.4482,Loss D: -0.0390
Epoch 5, Loss G: -0.0189,Loss D:  0.3223
Epoch 6, Loss G: -0.9843,Loss D:  0.3450
Epoch 7, Loss G: -0.7192,Loss D:  0.4247
Epoch 8, Loss G: -0.1643,Loss D:  0.0947
Epoch 9, Loss G: -0.7898,Loss D: -0.3226
Epoch 10, Loss G: -1.0775,Loss D: -0.2917
#END#"""

def test_extract_loss_info_from_stdout(input_str):
    expected_dict = {
        'SD0Q1_0': pd.DataFrame({'Epoch': [1, 2], 'Loss_G': [-0.0437, -0.1434], 'Loss_D': [-0.5837, 0.3780]}),
        'SD0Q1_1': pd.DataFrame({'Epoch': [1, 2], 'Loss_G': [-0.0437, -0.1434], 'Loss_D': [-0.5837, 0.3780]}),
        'SD0Q2_0': pd.DataFrame({'Epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Loss_G': [-0.0437, -0.1434, -0.1649, -0.4482, -0.0189, -0.9843, -0.7192, -0.1643, -0.7898, -1.0775], 'Loss_D': [-0.5837, 0.3780, 0.0226, -0.0390, 0.3223, 0.3450, 0.4247, 0.0947, -0.3226, -0.2917]})
    }
    result_dict = extract_loss_info_from_stdout(input_str)

    assert expected_dict.keys() == result_dict.keys(), \
        f"The dataset id's are not correct, Expected: {expected_dict.keys}, Actual: {result_dict.keys}"
        
    # Compare the dictionaries using numpy and assert_allclose
    for key in expected_dict.keys():
        assert np.allclose(result_dict[key].values, expected_dict[key].values)

def test_clean_and_convert_dict():

    # Test with the given example
    input_dict = {
        'C': '2.506',
        'class_weight': '{}',
        'dual': 'False',
        'fit_intercept': 'True',
        'intercept_scaling': '1',
        'l1_ratio': 'None',
        'max_iter': '1000',
        'multi_class': 'auto',
        'n_jobs': '-1',
        'penalty': 'l2',
        'random_state': '202',
        'solver': 'lbfgs',
        'tol': '0.0001',
        'verbose': '0',
        'warm_start': 'False',
    }
    expected = {
        'C': 2.506,
        'dual': False,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'l1_ratio': None,
        'max_iter': 1000,
        'multi_class': 'auto',
        'n_jobs': -1,
        'penalty': 'l2',
        'random_state': 202,
        'solver': 'lbfgs',
        'tol': 0.0001,
        'verbose': 0,
        'warm_start': False,
    }

    output = convert_and_clean_dict(input_dict)

    assert expected == output, f"The ouput is not equal to the expected values, \nExpected:\n{expected}\nOutput:\n{output}"