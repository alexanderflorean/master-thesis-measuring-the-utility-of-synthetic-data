import pandas as pd
import pytest

from src.utils import get_categorical_indicies

def test_get_categorical_indices():
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [1.1, 2.2, 3.3],
        'D': [True, False, True],
        'E': ['x', 'y', 'z']
    })

    metadata = {
        'fields': {
            'A': {'type': 'numeric'},
            'B': {'type': 'categorical'},
            'C': {'type': 'numeric'},
            'D': {'type': 'boolean'},
            'E': {'type': 'categorical'}
        }
    }

    expected_indices = [1, 3, 4]
    actual_indices = get_categorical_indicies(data, metadata)

    assert expected_indices == actual_indices

