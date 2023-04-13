import numpy as np

class Grids:

    @staticmethod
    def get_tuning_grid(model_name):
        model_name = model_name.lower()
        if model_name == 'lr':
            return Grids.logistic_regression_grid()
        elif model_name == 'knn':
            return Grids.k_nearest_neighbor_grid()
        elif model_name == 'nb':
            return Grids.naive_bayes_grid()
        #elif model_name == 'svm': ###For some reason, pycaret returns StochasticGradientDecentClassifier
        #    return Grids.support_vector_machines_grid()
        elif model_name == 'rbfsvm':
            return Grids.svm_rbf_kernel_grid()
        elif model_name == 'mlp':
            return Grids.multilayer_perceptron_grid()
        elif model_name == 'gbc':
            return Grids.gradient_boosting_classifier_grid()
        elif model_name == 'rf':
            return Grids.random_forest_grid()
        else:
            raise ValueError(f"Unknown model name '{model_name}'. Available options: 'lr', 'knn', 'nb', 'svm', 'mlp', 'gbc', 'rf'")

    @staticmethod
    def logistic_regression_grid():
        return {
            'C': np.logspace(-4, 4, 20).tolist(),
            'penalty': ['l1', 'l2' ],
            'solver': [ 'liblinear', 'saga'],
        }

    @staticmethod
    def k_nearest_neighbor_grid():
        return {
            'n_neighbors': list(range(1, 41)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
        }

    @staticmethod
    def naive_bayes_grid():
        return {
            'var_smoothing': np.logspace(-10, 0, 20).tolist()
        }

    @staticmethod
    def support_vector_machines_grid():
        return {
            'C': np.logspace(-4, 4, 20).tolist(),
            'kernel': ['linear', 'poly', 'sigmoid', 'rbf'],
            'degree': [2, 3, 4, 5],
            'coef0': [0.0, 0.1, 0.5, 1.0],
            'shrinking': [True, False]
        }

    @staticmethod
    def svm_rbf_kernel_grid():
        return {
            'C': np.logspace(-4, 4, 20).tolist(),
            'kernel': ['linear', 'poly', 'sigmoid', 'rbf'],
            'degree': [2, 3, 4, 5],
            'coef0': [0.0, 0.1, 0.5, 1.0],
            'shrinking': [True, False]
        }

    @staticmethod
    def multilayer_perceptron_grid():
        return {
            'hidden_layer_size_0': (50,), 
            'hidden_layer_size_1': (100,), 
            'hidden_layer_size_2': (250,), 
            'hidden_layer_size_3': (500,),
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
                1 ],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        }

    @staticmethod
    def gradient_boosting_classifier_grid():
        return {
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }

    @staticmethod
    def random_forest_grid():
        return {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

