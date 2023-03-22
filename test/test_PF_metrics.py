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
    assert 0 <= propensities['score'].min() <= 1, "Propensity scores out of range (0 to 1)"
    assert 0 <= propensities['score'].max() <= 1, "Propensity scores out of range (0 to 1)"


def test_pmse():
    pmse_score = pmse(original_df.copy(), synthetic_df.copy())
    assert 0 <= pmse_score , "Negative pMSE value"
    if (len(original_df) == len(synthetic_df)): 
        assert pmse_score <= 0.5, "pMSE value larger than 0.5, when original and synthetic datasets are of the same size."


def test_s_pmse():
    s_pmse_score = s_pmse(original_df.copy(), synthetic_df.copy())
    assert isinstance(s_pmse_score, float), "S_pMSE result is not a float"


def test_standardize_select_columns():
    # Prepare sample data
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [12, 24, 36, 48, 60],
        'C': [100, 200, 300, 400, 500]
    })

    # Test standardizing all columns
    standardized_data = standardize_select_data(data, [])
    print(standardized_data)
    assert np.isclose(standardized_data.mean(), 0).all()
    assert np.isclose(standardized_data.std(), 1).all()

    # Test standardizing specific columns and excluding one
    standardized_data = standardize_select_data(data, [1])
    assert np.isclose(standardized_data['A'].mean(), 0)
    assert np.isclose(standardized_data['A'].std(), 1)
    assert np.isclose(standardized_data['C'].mean(), 0)
    assert np.isclose(standardized_data['C'].std(), 1)
    assert not np.isclose(standardized_data['B'].mean(), 0)
    assert not np.isclose(standardized_data['B'].std(), 1)

    # Test standardizing with an invalid index
    try:
        standardized_data = standardize_select_data(data, [3])
    except IndexError:
        assert True
    else:
        assert False

    # Test with an empty DataFrame
    empty_data = pd.DataFrame()
    standardized_empty_data = standardize_select_data(empty_data, [])
    assert standardized_empty_data.empty