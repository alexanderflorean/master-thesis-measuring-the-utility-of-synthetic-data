import sys
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.tree import DecisionTreeClassifier

from src.PF_metrics import *

np.random.seed(42)

# Generate test data
original_data = np.random.normal(loc=0, scale=1, size=(100, 3))
original_df = pd.DataFrame(original_data, columns=['A', 'B', 'C'])

synthetic_data = np.random.normal(loc=0.5, scale=1, size=(100, 3))
synthetic_df = pd.DataFrame(synthetic_data, columns=['A', 'B', 'C'])


def test_compute_propensities():
    propensities = compute_propensity(original_df.copy(), synthetic_df.copy())
    #assert len(propensities) == 200, "Incorrect length of propensities array"
    assert 0 <= propensities.min() <= 1, "Propensity scores out of range (0 to 1)"
    assert 0 <= propensities.max() <= 1, "Propensity scores out of range (0 to 1)"


def test_pMSE():
    pMSE_score = pMSE(original_df.copy(), synthetic_df.copy())
    assert 0 <= pMSE_score , "Negative pMSE value"
    if (len(original_df) == len(synthetic_df)): 
        assert pMSE_score <= 0.5, "pMSE value larger than 0.5, when original and synthetic datasets are of the same size."


def test_S_pMSE():
    S_pMSE_score = S_pMSE(original_df.copy(), synthetic_df.copy())
    assert isinstance(S_pMSE_score, float), "S_pMSE result is not a float"


def test_standardize_select_columns():
    # Example usage:
    test_input = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40],
        'C': [100, 200, 300, 400]
    })
    expected = pd.DataFrame({
    'A': [-1.341641, -0.447214, 0.447214, 1.341641],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
    })


    exclude_indices = [1, 2]  # Do not standardize column B and C
    output_df = standardize_select_columns(test_input, exclude_indices)

    assert output_df.shape == expected.shape, f"The function returned wrong shape, \nexpected: {expected.shape}, \nouput: {output_df.shape}"

    assert expected.columns.equals(output_df.columns), f"The function returned wrong columns, \nexpected: {expected.columns}, \noutput: {output_df.columns}"

    float_columns = output_df.select_dtypes(include=[np.float64, np.float32]).columns
    non_float_columns = output_df.columns.difference(float_columns)

    float_close = np.isclose(output_df[float_columns], expected[float_columns]).all().all()

    assert float_close, "Output from function generated wrong float values"

    non_float_equal = output_df[non_float_columns].equals(expected[non_float_columns])
    assert non_float_equal, "Ouput from the function changed non float columns"
