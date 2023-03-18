import pandas as pd
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


def compute_propensity(original_data, synthetic_data):
    """     Uses the CART model from sklearn to compute the popensity 
            of predicting the synthetic data, i.e. the pobability that 
            the model predicts the synthetic data.

                The method assumes the original and synthetic data has 
                the same features.

            In:
                original_data:  pandas.DataFrame
                synthetic_data: pandas.DataFrame
            Out: 
                List of probabilities for predicting the samples are synthetic data
    """
    
    # Add target label 'S', i.e. if the sample is synthetic 
    original_data['S'] = 0
    synthetic_data['S'] = 1
    
    # Combine original_data and synthetic_data
    combined_data = pd.concat([original_data, synthetic_data], axis=0)
    Z = combined_data.drop(columns='S')   # remove target label
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(Z, 
                                                        combined_data['S'], 
                                                        test_size=0.3, 
                                                        random_state=42
                                                        )

    # fit CART model and compute propensity scores
    clf = DecisionTreeClassifier()
    ''' TODO: set default settings, however need to double check
            test_size=0.25,     random_state=42,
            criterion='gini',   splitter='best', 
            max_depth=None,     min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            max_features=None,  max_leaf_nodes=None, 
            min_impurity_split=None, cp_alpha=0.0,
            class_weight=None,   presort='deprecated', 
            min_impurity_decrease=0.0 
    '''

    clf.fit(X_train, y_train)

    # Extract probabilities for class 1 (synthetic) on X_test datapoints
    score = clf.predict_proba(X_test)[:, 1] 
    
    return score


def pMSE(original_data, synthetic_data):
    # Calculate the propensity mean-squared error
    
    n_o = original_data.shape[0]   # number of samples in original data
    n_s = synthetic_data.shape[0]  # number of samples in synthetic data
    N = n_o + n_s
    c = n_s / N
    
    propensity = compute_propensity(original_data, synthetic_data)
    
    pMSE_score = sum((propensity - c) ** 2 ) / N
    
    return pMSE_score


def S_pMSE(original_data, synthetic_data):
    # Calculate the standardized propensity mean-squared error
    
    # Get variables
    k = original_data.shape[1]  # number of predictors of the combined dataset (i.e. target 'S')
    n_o = original_data.shape[0]     # number of samples in original data
    n_s = synthetic_data.shape[0]    # number of samples in synthetic data
    N = n_o + n_s
    c = n_s / N
    
    pMSE_score = pMSE(original_data, synthetic_data)
    
    # simulated results for the expected and standard deviation  pMSE value
    sim_pMSE = (k-1) * (1 - c) ** 2 * c / N
    std_pMSE = sqrt(2 * (k-1)) * (1 - c) ** 2 * c / N
    
    S_pMSE_score = (pMSE_score - sim_pMSE) / std_pMSE
    
    return S_pMSE_score


def calculate_cluster_weight(weights, 
                             cluster_id, 
                             target_cluster_data_count, 
                             cluster_data_count, 
                             total_data_count):

    """Calculate the weight for a given cluster."""

    if weights is None:
        return 1
    elif weights == "approx_std_err":
        # computes the percentage of the target_data_count over the total cluster count
        percentage = target_cluster_data_count / cluster_data_count
        return np.sqrt((percentage * (1 - percentage)) / total_data_count)
    else:
        return weights[cluster_id]


def cluster_analysis_metric(original_data, 
                            synthetic_data, 
                            num_clusters, 
                            weights="approx_std_err"):
    """
    Calculate the log cluster metric for the given original and synthetic datasets.
    
    Args:
        original_data (DataFrame): The original dataset.
        synthetic_data (DataFrame): The synthetic dataset.
        num_clusters (int): The number of clusters to use in the KMeans clustering.
        weights (Union[str, List[float]]): The type of weights to use or a list of weights.

            None: sets all cluster weights to 1.

            "approx_std_err": computes the approximate standard error for the percentage of 
            the number of synthetic data samples in the current cluster.
        
    Returns:
        float: The cluster analysis metric.
    """

    combined_data = pd.concat([original_data, synthetic_data], axis=0)

    scaled_combined_data = StandardScaler().fit_transform(combined_data)
    
    # Perform clustering on the scaled combined data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(scaled_combined_data)
    
    cluster_labels = kmeans.labels_
    
    original_data_count = original_data.shape[0]    # number of samples in original data
    synthetic_data_count = synthetic_data.shape[0]  # number of samples in synthetic data
    total_data_count = original_data_count + synthetic_data_count

    constant_c = original_data_count / (original_data_count + synthetic_data_count)

    Uc = 0  # initialize Uc

    for cluster_id in range(num_clusters):

        original_cluster_data_count = np.sum(cluster_labels[:original_data_count] == cluster_id)
        synthetic_cluster_data_count = np.sum(cluster_labels[original_data_count:] == cluster_id)

        total_cluster_data_count = original_cluster_data_count + synthetic_cluster_data_count

        # TODO: question, should I use the target_count for synthetic or original data count here?
        weight = calculate_cluster_weight(weights=weights, 
                                  cluster_id=cluster_id, 
                                  target_cluster_data_count=original_cluster_data_count, 
                                  cluster_data_count=total_cluster_data_count, 
                                  total_data_count=total_data_count)

        Uc += weight * ((original_cluster_data_count / total_cluster_data_count) - constant_c)** 2

    # Normalize Uc by the number of clusters
    Uc /= num_clusters
    return Uc

