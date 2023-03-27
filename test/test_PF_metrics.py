import numpy as np
import pandas as pd

from math import sqrt
from sklearn.datasets import make_blobs

import sys, os
# quick fix for ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')
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

def test_cluster_metric():

    # Create sample original_data DataFrame
    original_data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    original_data = pd.DataFrame(original_data, columns=['A', 'B'])
    original_data['S'] = 0

    # Create sample synthetic_data DataFrame
    synthetic_data, _ = make_blobs(n_samples=100, centers=3, random_state=24)
    synthetic_data = pd.DataFrame(synthetic_data, columns=['A', 'B'])
    synthetic_data['S'] = 1

    # Create metadata dictionary
    metadata = {
        'columns': {
            'A': {'type': 'numeric'},
            'B': {'type': 'numeric'},
            'S': {'type': 'boolean'},
        }
    }

    # Call cluster_metric with sample data and metadata
    num_clusters = 3
    actual_metric = cluster_metric(original_data, synthetic_data, num_clusters, metadata, random_state=42)

    # Compare the result with an expected metric value
    expected_metric = 0.02  # Replace with an expected metric value based on your sample data
    np.testing.assert_almost_equal(actual_metric, expected_metric, decimal=3)


def test_standardize_select_columns():
    # Prepare sample data
    input_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [True, False, True, False, False],
        'C': [33, 24, 16, 58, 60],
        'D': [1, 1, 2, 0, 0],
        'E': [5, 382, 389, 411, 502],
    })
    categorical_columns = [1, 3]

    expected = input_data.copy()

    scaler = StandardScaler()

    s_data = input_data.copy()
    s_data = s_data.drop(columns=s_data.columns[categorical_columns])

    scaled = scaler.fit_transform(s_data)

    expected['A'] = scaled.T[0]
    expected['C'] = scaled.T[1]
    expected['E'] = scaled.T[2]

    # Test standardizing all columns
    actual = standardize_select_columns(input_data, categorical_columns)

    #print("\nExpected: ")
    #print(expected)

    #print("\nOutput: ")
    #print(actual)

    assert np.isclose(expected['A'], actual['A']).all()
    assert np.isclose(expected['B'], actual['B']).all()
    assert np.isclose(expected['C'], actual['C']).all()
    assert np.isclose(expected['D'], actual['D']).all()
    assert np.isclose(expected['E'], actual['E']).all()

    # Test standardizing with invalid indices
    try:
        standardized_data = standardize_select_columns(input_data, [13, 20])
    except IndexError:
        assert True
    else:
        assert False