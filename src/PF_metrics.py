from math import sqrt

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sdmetrics.single_table import (BNLikelihood, BNLogLikelihood,
                                    GMLogLikelihood)
from sdmetrics.single_table.efficacy import (BinaryMLPClassifier,
                                             MulticlassMLPClassifier)
from sdmetrics.single_table.multi_column_pairs import (ContinuousKLDivergence,
                                                       DiscreteKLDivergence)
from sdmetrics.single_table.multi_single_column import CSTest, KSComplement
from sdmetrics.utils import (get_columns_from_metadata,
                             get_type_from_column_meta)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import get_categorical_indices, timefunction


### Start - pMSE & S_pMSE
def compute_propensity(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    classifier=LogisticRegression(),
    random_state=None,
) -> dict:
    """
    Uses Logistic Regression from sklearn to compute the propensity
    of predicting the synthetic data, i.e. the probability that
    the model predicts the synthetic data.

    The method assumes the original and synthetic data has
    the same features.

    Parameters:
        -----------

        original_data: pandas.DataFrame
        synthetic_data: pandas.DataFrame
        classifier: scikit-learn classifier
        random_state: int, optional, default: None
            Controls the random state for train_test_split


    Returns:
        --------

        Dictionary with the following:
            'score': List of probabilities for predicting the samples are synthetic data
            'no': number of original data samples in the test data
            'ns': number of synthetic data samples in the test data
    """
    # TODO: consider standardizing values and encoding the categorical predictors
    original_data = original_data.copy()
    synthetic_data = synthetic_data.copy()

    # Add target label 'S', i.e. if the sample is synthetic
    original_data["S"] = 0
    synthetic_data["S"] = 1

    # Combine original_data and synthetic_data
    combined_data = pd.concat([original_data, synthetic_data], axis=0)
    Z = combined_data.drop(columns="S")  # remove target label

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        Z, combined_data["S"], test_size=0.3, random_state=random_state
    )

    n_o_test = sum(y_test == 0)  # number of original samples in the test data
    n_s_test = sum(y_test == 1)  # number of synthetic samples in the test data

    # fit Logistic Regression model and compute propensity scores
    classifier.fit(X=X_train, y=y_train)

    # Extract probabilities for class 1 (synthetic) on X_test datapoints
    score = classifier.predict_proba(X_test)[:, 1]

    return {"score": score, "no": n_o_test, "ns": n_s_test}


@timefunction
def pmse(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    classifier=LogisticRegression(),
) -> float:
    """
    Calculate the propensity mean-squared error.
    Algorithm implemented as described in DOI: 10.1111/rssa.12358

    Parameters:
        -----------

        original_data: pandas.DataFrame
        synthetic_data: pandas.DataFrame
        classifier: (optional) scikit-learn classifier,
                    default=LogisticRegression

    Returns:
        --------

        float: pMSE score
    """
    prop_dict = compute_propensity(original_data, synthetic_data, classifier)

    propensity = prop_dict["score"]

    n_o = prop_dict["no"]  # number of samples from original data
    n_s = prop_dict["ns"]  # number of samples from synthetic data
    N = n_o + n_s
    c = n_s / N  # proportion of # synthetic samples in the test data

    pmse_score = sum((propensity - c) ** 2) / N

    return pmse_score


@timefunction
def s_pmse(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    classifier=LogisticRegression(),
) -> float:
    """
    Calculate the standardized propensity mean-squared error
        Algorithm implemented as described in DOI: 10.1111/rssa.12358

    Parameters:
        -----------

        original_data: pandas.DataFrame
        synthetic_data: pandas.DataFrame
        classifier: (optional) scikit-learn classifier,
                    default=LogisticRegression

    Returns:
        --------

        float: standardized pMSE score
    """
    # TODO: neeed to double check implementation

    # number of predictors of the combined dataset (i.e. target 'S')
    k = original_data.shape[1] 

    prop_dict = compute_propensity(original_data, synthetic_data, classifier)

    propensity = prop_dict["score"]

    n_o = prop_dict["no"]  # number of samples from original data
    n_s = prop_dict["ns"]  # number of samples from synthetic data
    N = n_o + n_s
    c = n_s / N  # proportion of # synthetic samples in the test data

    pmse_score = sum((propensity - c) ** 2) / N

    # simulated results for the expected and standard deviation  pMSE value
    sim_pmse = (k - 1) * (1 - c) ** 2 * c / N
    std_pmse = sqrt(2 * (k - 1)) * (1 - c) ** 2 * c / N

    s_pmse_score = (pmse_score - sim_pmse) / std_pmse

    return s_pmse_score


### End - pMSE & S_pMSE


### Start - Cluster analysis
def calculate_cluster_weight(
    target_cluster_data_count: int,
    cluster_data_count: int,
    total_data_count: int,
    weight_type: str = "count",
) -> float:
    """
    Calculate the approximate standard error for the percentage of the
    target count in the provided cluster data.

    Parameters:
        -----------

        target_cluster_data_count: (int)
            The number of target data points in the cluster

        cluster_data_count: (int)
            The total number of data points in the cluster

        total_data_count: (int)
            The total number of data points across all clusters

        weight_type: (str), default="count"
            "approx": The approximate standard error for the given cluster over all samples
            "count": The the number of samples in the cluster divided by the total number of samples in the dataset

    Returns:
        --------

        float: the computed weight
    """
    if weight_type == "approx":
        percentage = target_cluster_data_count / cluster_data_count
        return np.sqrt((percentage * (1 - percentage)) / total_data_count)
    elif weight_type == "count":
        return cluster_data_count  # / total_data_count
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")


def standardize_select_columns(
    data: pd.DataFrame, indices_to_exclude: list
) -> pd.DataFrame:
    """
    Standardizes a pandas DataFrame with the exception of specified columns.

    Parameters:
        -----------

        data (pd.DataFrame): The input DataFrame to standardize.
        indices_to_exclude (list): A list of column indices to exclude from standardization.

    Returns:
        --------

        pd.DataFrame: The standardized DataFrame with the exception for specified columns, which are unmodified.

    """
    data = data.copy()
    scaler = StandardScaler()

    column_indices = np.arange(data.shape[1])

    # Check for indices out of bound
    diff = set(indices_to_exclude) - set(column_indices)
    if diff:
        raise IndexError(
            f"indicies_to_exclude contains index not in the columns of the dataset: {', '.join(map(str , diff))}"
        )

    columns_to_standardize = np.setdiff1d(column_indices, indices_to_exclude)
    data.iloc[:, columns_to_standardize] = scaler.fit_transform(
        data.iloc[:, columns_to_standardize]
    )

    return data


@timefunction
def cluster_metric(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    num_clusters: int,
    metadata: dict,
    random_state=None,
) -> float:
    """
    Calculate the cluster analysis metric for the given original and synthetic datasets.

        Algorthim implemented as described in DOI: 10.29012/jpc.v1i1.568

    Parameters:
        -----------

        original_data:  pandas.DataFrame
            The original dataset.

        synthetic_data: pandas.DataFrame
            The synthetic dataset.

        num_clusters:   integer
            The number of clusters to use in the clustering.

        categorical_columns: list[int]
            List of indices of columns that contain categorical data

    Returns:
        --------

        float: The cluster analysis metric.
    """

    combined_data = pd.concat(
        [original_data, synthetic_data], axis=0, copy=True, ignore_index=True
    )

    categorical_columns = get_categorical_indices(original_data, metadata)

    assert (
        max(categorical_columns) <= combined_data.shape[1]
    ), f"Categorical index: {max(categorical_columns)} is larger the than number of columns: {combined_data.shape[1]} in the data"

    if categorical_columns == []:
        # Contains only numerical cols, standardize the combined data
        scaled_combined_data = StandardScaler().fit_transform(combined_data)
        # Cluster the scaled data with sklearn.KMeans()
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(
            scaled_combined_data
        )  # expects shape (n_samples, n_features)
        cluster_labels = kmeans.labels_  # returns the

    elif len(categorical_columns) == combined_data.shape[1]:
        # TODO: use k-modes
        raise NotImplementedError(
            "Case: all columns are categorical, solution: use k-modes, however it is not implmented yet."
        )

    else:
        # Perform clustering on the combined data using KPrototypes, it already encodes categorical attributes
        scaled_combined_data = standardize_select_columns(
            combined_data, indices_to_exclude=categorical_columns
        )
        # TODO: remove logg
        # print(
        #    f"num samples data: {len(scaled_combined_data)}, num_klusters:{num_clusters}"
        #)
        kproto = KPrototypes(
            n_clusters=num_clusters, init="Cao", random_state=random_state, n_jobs=1
        )
        # Kprottypes expects data with variables as columns, which is why it is transposed
        #print(scaled_combined_data.shape)
        #print(type(scaled_combined_data))
        kproto.fit(scaled_combined_data, categorical=categorical_columns)

        cluster_labels = (
            kproto.labels_
        )  # returns the labels with same indices as the learned dataset

    original_data_count = original_data.shape[0]  # number of samples in original data
    synthetic_data_count = synthetic_data.shape[0]  # number of samples in synthetic data
    total_data_count = original_data_count + synthetic_data_count

    constant_c = original_data_count / (original_data_count + synthetic_data_count)

    Uc = 0  # initialize Uc

    for cluster_id in range(num_clusters):
        original_cluster_data_count = np.sum(
            cluster_labels[:original_data_count] == cluster_id
        )
        synthetic_cluster_data_count = np.sum(
            cluster_labels[original_data_count:] == cluster_id
        )

        total_cluster_data_count = (
            original_cluster_data_count + synthetic_cluster_data_count
        )

        weight = calculate_cluster_weight(
            target_cluster_data_count=original_cluster_data_count,
            cluster_data_count=total_cluster_data_count,
            total_data_count=total_data_count,
            weight_type="count",
        )

        Uc += (
            weight
            * ((original_cluster_data_count / total_cluster_data_count) - constant_c)
            ** 2
        )

    Uc /= num_clusters
    return Uc


### End - Cluster analysis


### Start - Likehood measures:  Looks at the likelihood of the synthetic data belonging to the real data.
@timefunction
def BNLikelihood_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    """Uses Bayesian Network to learn the distribution of the real data to then average
    likelihood for all samples on wether they belong to the real data, range[0,1], where 0 means it doesn't
    belong to the real data and 1 that it does.

    Meant for boolean, categorical data and ignores the other incompatible column types,
    NOTE: this metric does not accept missing values.
    For more details see:
        https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/data-likelihood/bnlikelihood

    """
    return BNLikelihood.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )


@timefunction
def BNLogLikelihood_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    """Returns the log of BNLikelihood
    Range[-inf, 1]
    Meant for boolean, categorical data and ignores the other incompatible column types,
    NOTE: this metric does not accept missing values.
    For more details see:
        https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/data-likelihood/bnloglikelihood
    """
    return BNLogLikelihood.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )


@timefunction
def GMLogLikelihood_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    """Uses multiple Gaussian mixtures models to learn the distribution of the real data to then returns
    the average likelihood for all samples on wether they belongs to the real data or not.
    Range[-inf,+inf], where -inf means it doesn't belong to the real data and +inf that it does.
    Meaning, the larger the value the more likely the sample/s belong to the real data and
    the smaller means the opposite.

    Meant for continous, numerical data and ignores the other incompatible column types,
    NOTE: this metric does not accept missing values.
    For more details see:
        https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/data-likelihood/gmlikelihood

    """
    return GMLogLikelihood.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )


### End - Likehood measures


### Start - Difference in Empirical distributions type measures
@timefunction
def ContinousKLDivergence_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    return ContinuousKLDivergence.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )


@timefunction
def DiscreteKLDivergence_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    return DiscreteKLDivergence.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )


@timefunction
def KSComplement_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/kscomplement
    return KSComplement.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )


@timefunction
def CSTest_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/cstest
    # Might be better to use the TVComplement measure, as CSTest is mentioned to have some issues
    return CSTest.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )


### End - Difference in Empirical distributions type measures


### Start - Cross-Classification measure
@timefunction
def CrossClassification_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
) -> float:
    # get categorical columns,
    # identify if they are binary or multiclass,
    # run respective MLEfficacy algorithm, returns the F1-score
    # TODO/consideration, the measure produces different results everytime it is run, should it therefore
    # multiple times then calculate the average or use random-state seed?

    results = []
    columns = get_columns_from_metadata(metadata)

    for col in columns:
        col_type = get_type_from_column_meta(columns[col])
        if col_type == "categorical":
            target_data = original_data[col]
            uniques = target_data.nunique()
            if uniques == 2:
                # removes the target label from the training?
                # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/ml-efficacy-single-table/binary-classification
                efficacy = BinaryMLPClassifier.compute(
                    test_data=original_data,
                    train_data=synthetic_data,
                    metadata=metadata,
                    target=col,
                )
            else:
                # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/ml-efficacy-single-table/multiclass-classification
                efficacy = MulticlassMLPClassifier.compute(
                    test_data=original_data,
                    train_data=synthetic_data,
                    metadata=metadata,
                    target=col,
                )

            results.append(efficacy)

    return np.mean(results)


### End - Cross-Classification measure


### Computes all metrics defined above
def compute_all_pf_measures(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata: dict,
    SD_id: str,
) -> pd.DataFrame:
    # get number of clusters, using the combined number of samples in the synthetic & original data,
    # and round it to an integer
    one_percent = 0.01
    five_percent = 0.05
    ten_percent = 0.1

    k_1 = round((original_data.shape[0] + synthetic_data.shape[0]) * one_percent)
    k_5 = round((original_data.shape[0] + synthetic_data.shape[0]) * five_percent)
    k_10 = round((original_data.shape[0] + synthetic_data.shape[0]) * ten_percent)

    measures = {}

    measures["DatasetName"] = SD_id
    measures["SDG"] = SD_id.split('_')[0]

    result = pmse(original_data=original_data, synthetic_data=synthetic_data)
    measures["pMSE"] = result["score"]
    measures["pMSE_time"] = result["time"]

    result = s_pmse(original_data=original_data, synthetic_data=synthetic_data)
    measures["s_pMSE"] = result["score"]
    measures["s_pMSE_time"] = result["time"]

    result = cluster_metric(
        original_data=original_data,
        synthetic_data=synthetic_data,
        num_clusters=k_1,
        metadata=metadata,
    )
    measures["Cluster_1"] = result["score"]
    measures["Cluster_1_time"] = result["time"]

    result = BNLogLikelihood_metric(
        original_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )
    measures["BNLogLikelihood"] = result["score"]
    measures["BNLogLikelihood_time"] = result["time"]

    result = GMLogLikelihood_metric(
        original_data=original_data, synthetic_data=synthetic_data, metadata=metadata
    )
    measures["GMLogLikelihood"] = result["score"]
    measures["GMLogLikelihood_time"] = result["time"]

    result = ContinousKLDivergence_metric(original_data, synthetic_data, metadata)
    measures["ContinousKLDivergence"] = result["score"]
    measures["ContinousKLDivergence_time"] = result["time"]

    result = DiscreteKLDivergence_metric(original_data, synthetic_data, metadata)
    measures["DiscreteKLDivergence"] = result["score"]
    measures["DiscreteKLDivergence_time"] = result["time"]

    result = KSComplement_metric(original_data, synthetic_data, metadata)
    measures["KSComplement"] = result["score"]
    measures["KSComplement_time"] = result["time"]

    result = CSTest_metric(original_data, synthetic_data, metadata)
    measures["CSTest"] = result["score"]
    measures["CSTest_time"] = result["time"]

    result = CrossClassification_metric(original_data, synthetic_data, metadata)
    measures["CrossClassification"] = result["score"]
    measures["CrossClassification_time"] = result["time"]

    results_df = pd.DataFrame(data=measures, index=[0])
    return results_df