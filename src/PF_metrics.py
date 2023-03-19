import pandas as pd
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes

### Start - pMSE & S_pMSE
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
    # TODO: consider standardizing values and encoding the categorical predictors
    
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
                                                        random_state=42)

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
    ''' 
    Calculate the propensity mean-squared error
        Algorithm implemented as described in DOI: 10.1111/rssa.12358
    '''
    
    n_o = original_data.shape[0]   # number of samples in original data
    n_s = synthetic_data.shape[0]  # number of samples in synthetic data
    N = n_o + n_s
    c = n_s / N
    
    propensity = compute_propensity(original_data.copy(), synthetic_data.copy())
    
    pMSE_score = sum((propensity - c) ** 2 ) / N
    
    return pMSE_score


def S_pMSE(original_data, synthetic_data):
    ''' 
    Calculate the standardized propensity mean-squared error
        Algorithm implemented as described in DOI: 10.1111/rssa.12358
    '''
    
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
### End - pMSE & S_pMSE


### Start - Cluster analysis
def calculate_cluster_weight(weights, 
                             cluster_id, 
                             target_cluster_data_count, 
                             cluster_data_count, 
                             total_data_count):

    """ Calculate the weight for a given cluster """

    if weights is None:
        return 1
    elif weights == "approx_std_err":
        # computes the percentage of the target_data_count over the total cluster count
        percentage = target_cluster_data_count / cluster_data_count
        return np.sqrt((percentage * (1 - percentage)) / total_data_count)
    else:
        return weights[cluster_id]


def standardize_select_columns(data, indices_to_exclude):
    """
    Standardize a selection of columns in a dataset
        
        Args:
            
            data: pandas.DataFrame

            indices_to_exclude: list[int]
                List of indices of columns in the dataset to not standardize

        Returns:
            Dataset with specified columns to standardize

    """
    scaler = StandardScaler()
    column_indices = np.arange(data.shape[1])

    columns_to_standardize = np.setdiff1d(column_indices, indices_to_exclude)
    
    data.iloc[:, columns_to_standardize] = scaler.fit_transform(data.iloc[:, columns_to_standardize])
    
    return data


def cluster_analysis_metric(original_data, 
                            synthetic_data, 
                            num_clusters, 
                            categorical_columns=None,
                            weights="approx_std_err",
                            random_state=42):
    """
    Calculate the cluster analysis metric for the given original and synthetic datasets.

        Algorthim implemented as described in DOI: 10.29012/jpc.v1i1.568
    
    Args:
        original_data:  pandas.DataFrame
            The original dataset.

        synthetic_data: pandas.DataFrame
            The synthetic dataset.

        num_clusters:   integer
            The number of clusters to use in the clustering.

        categorical_columns: list[int]
            List of indices of columns that contain categorical data

        weights:    Union[str, List[float]]
            The type of weights to use or a list of weights.

                None: sets all cluster weights to 1.

                "approx_std_err": computes the approximate standard error for the 
                    percentage of the number of synthetic data samples in the 
                    current cluster.

                List[float]: the weights for each cluster, following must be true: 
                    len(List[float]) == num_clusters
        
    Returns:
        float: The cluster analysis metric.
    """

    combined_data = pd.concat([original_data, synthetic_data], axis=0)

    if(categorical_columns == None):
        # Contains only numerical cols, standardize the combined data
        scaled_combined_data = StandardScaler().fit_transform(combined_data)
        # Cluster the scaled data with sklearn.KMeans()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(scaled_combined_data)
        cluster_labels = kmeans.labels_

    else:
        # Perform clustering on the combined data using KPrototypes, it already encodes categorical attributes
        # Standardize non categorical columns
        scaled_combined_data = standardize_select_columns(combined_data, indices_to_exclude=categorical_columns)
        kproto = KPrototypes(n_clusters=num_clusters, init='Cao', random_state=42).fit(scaled_combined_data, categorical=categorical_columns)
        cluster_labels = kproto.labels_

    
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

    Uc /= num_clusters
    return Uc
### End - Cluster analysis
