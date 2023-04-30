import numpy as np
from optuna.distributions import IntUniformDistribution

class Grids:

    @staticmethod
    def get_tuning_grid(model_name, ml_engine='sklearn'):
        model_name = model_name.lower()
        ml_engine = ml_engine.lower()
        
        if ml_engine == "cuml":
            if model_name == 'lr':
                return Grids.logistic_regression_cuml_grid()
            elif model_name == 'knn':
                return Grids.k_nearest_neighbor_cuml_grid()
            elif model_name == 'nb':
                return Grids.naive_bayes_cuml_grid()
            elif model_name == 'svm' or model_name=='rbfsvm':
                return Grids.svm_cuml_grid()
            elif model_name == 'rf':
                return Grids.random_forest_cuml_grid()
            else:
                raise ValueError(f"Unknown model name '{model_name}' for cuML. Available options: 'lr', 'knn', 'nb', 'svm', 'rf'")
        elif ml_engine == "sklearn":
            if model_name == 'lr':
                return Grids.logistic_regression_sklearn_grid()
            elif model_name == 'knn':
                return Grids.k_nearest_neighbor_sklearn_grid()
            elif model_name == 'svm' or model_name=='rbfsvm':
                return Grids.svm_sklearn_grid()
            elif model_name == 'mlp':
                return Grids.multilayer_perceptron_sklearn_grid()
            elif model_name == 'gbc':
                return Grids.gradient_boosting_classifier_sklearn_grid()
            elif model_name == 'rf':
                return Grids.random_forest_sklearn_grid()
            elif model_name == 'nb':
                return Grids.naive_bayes_sklearn_grid()
            else:
                raise ValueError(f"Unknown model name '{model_name}' for sklearn. Available options: 'lr', 'knn', 'svm', 'mlp', 'gbc', 'rf'")
        else:
            raise ValueError(f"Unknown ML engine '{ml_engine}'. Available options: 'cuml', 'sklearn'")
    
    # For cuML
    @staticmethod
    def logistic_regression_cuml_grid():
        return {
            'C': np.logspace(-4, 4, 20).tolist(),
            'penalty': ['l1', 'l2'],
        }
    
    # For sklearn
    @staticmethod
    def logistic_regression_sklearn_grid():
        return {
            'C': np.logspace(-4, 4, 20).tolist(),
            'penalty': ['l2', None],
        }
    
    # For cuML
    @staticmethod
    def k_nearest_neighbor_cuml_grid():
        return {
            'n_neighbors': list(range(1, 41)),
            'weights': ['uniform'],   # 'distance' not supported by cuML
            'metric': ['euclidean', 'manhattan', 'minkowski'],
        }
    
    # For sklearn
    @staticmethod
    def k_nearest_neighbor_sklearn_grid():
        return {
            'n_neighbors': list(range(1, 41)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }
    
    # For cuML
    @staticmethod
    def naive_bayes_cuml_grid():
        return {
            'alpha': np.logspace(-10, 0, 20).tolist(),
            'output_type': 'cudf',
        }
    
    # For sklearn
    @staticmethod
    def naive_bayes_sklearn_grid():
        return {
            'var_smoothing': np.logspace(-10, 0, 20).tolist(),
            'fit_prior': [True, False]
        }

    # For cuML
    @staticmethod
    def svm_cuml_grid():
        return {
            'C': np.logspace(-4, 4, 20).tolist(),
            'kernel': ['poly', 'sigmoid', 'rbf'],
            'degree': [2, 3, 4, 5, 6],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.1, 0.5, 1.0],
            'cache_size': 4096,
        }

    # For sklearn
    @staticmethod
    def svm_sklearn_grid():
        return {
            'C': np.logspace(-4, 4, 20).tolist(),
            'kernel': ['poly', 'sigmoid', 'rbf'],
            'degree': [2, 3, 4, 5, 6],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.1, 0.25, 0.5, 0,75, 1.0],
            'shrinking': [True, False],
            'max_iter': [100000, 100000]
        }

    # For sklearn
    @staticmethod
    def multilayer_perceptron_sklearn_grid():
        return {
            'hidden_layer_sizes': [(50,), (100,), (250,), (500,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': np.logspace(-4, 1, 20).tolist(),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        }

    # For sklearn
    @staticmethod
    def gradient_boosting_classifier_sklearn_grid():
        return {
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }

    # For cuML
    @staticmethod
    def random_forest_cuml_grid():
        return {
            'split_criterion': [0, 1],   # [0 = 'gini', 1 = 'entropy']
            'n_estimators': list(range(100, 500, 10)),
            'max_depth': list(range(10, 1000, 10)), #IntUniformDistribution(10, 1000, 10),
            'min_samples_split': [0.1, 0.25, 0.5, 0.75, 1.0],
            'min_samples_leaf': [0.1, 0.25, 0.5, 0.75, 0.9],
            'max_features': [1.0, 0.75, 0.5],   # alt 'log2'
            'n_bins': [128, 256, 512],
            'bootstrap': [True, False]
        }

    # For sklearn
    @staticmethod
    def random_forest_sklearn_grid():
        return {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'n_estimators': list(range(100, 500, 10)),
            'max_depth': list(range(10, 1000, 10)), #IntUniformDistribution(10, 1000, 10),
            'min_samples_split': [0.1, 0.25, 0.5, 0.75, 1.0],
            'min_samples_leaf': [0.25, 0.5, 0.75, 0.9],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'bootstrap': [True, False]
        }