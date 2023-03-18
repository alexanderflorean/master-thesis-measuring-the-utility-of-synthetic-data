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

