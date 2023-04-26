from cuml.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, LabelEncoder
from cuml.compose import ColumnTransformer
from cuml.pipeline import Pipeline
from cuml.metrics import accuracy_score
import cuml
import cudf
import cupy as cp

"""
This module is used for createing classifiers that utilize gpu,
it was found that the cpu-models from sklearn was inssuficient 
in the experiment.
"""
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder

class OrdinalEncoder(SklearnOrdinalEncoder):
    def __init__(self, categories=None):
        super().__init__(categories=categories)

    def fit(self, X, y=None):
        X_pd = X.to_pandas()
        return super().fit(X_pd, y)

    def transform(self, X):
        X_pd = X.to_pandas()
        X_transformed = super().transform(X_pd)
        return cudf.DataFrame(X_transformed, columns=X.columns, index=X.index)

    def inverse_transform(self, X_transformed):
        X_transformed_pd = X_transformed.to_pandas()
        X_original = super().inverse_transform(X_transformed_pd)
        return cudf.DataFrame(X_original, columns=X_transformed.columns, index=X_transformed.index)


class GPUClassifierPipeline:
    def __init__(self, classifier, numeric_features=None, categorical_features=None, ordinal_features=None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features

        transformers = []

        # Preprocessing for numeric features
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))

        # Preprocessing for categorical features
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(), categorical_features))

        # Preprocessing for ordinal features
        if ordinal_features:
            for ordinal_feature, categories in ordinal_features.items():
                transformers.append((ordinal_feature, OrdinalEncoder(categories=[categories]), [ordinal_feature]))

        preprocessor = ColumnTransformer(transformers=transformers)

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        
        # Initialize LabelEncoder
        self.lb = LabelBinarizer()

    def swap_classifier(self, new_classifier):
        self.pipeline.steps[-1] = ('classifier', new_classifier)

    def fit(self, X, y):
        self.lb.fit(y.astype(np.int32).values.ravel())
        y_transformed = self.lb.transform(y.astype(np.int32)).values.ravel()  # Convert y to int32 and flatten to 1D array
        
        self.pipeline.fit(X, y_transformed)

    def predict(self, X):
        y_pred_transformed = self.pipeline.predict(X)
        y_pred = self.lb.inverse_transform(y_pred_transformed)
        return y_pred
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_classifier(self):
        return self.pipeline.named_steps['classifier']
    
    def set_params(self, **params):
        self.pipeline.named_steps['classifier'].set_params(**params)
        
    def get_params(self):
        return self.pipeline.named_steps['classifier'].get_params()




def GPUModels(classifier_type):
    if classifier_type == 'lr':
        return cuml.linear_model.LogisticRegression()
    elif classifier_type == 'knn':
        return cuml.neighbors.KNeighborsClassifier()
    elif classifier_type == 'nb':
        return cuml.naive_bayes.MultinomialNB()
    elif classifier_type == 'rbfsvm':
        return cuml.svm.SVC(kernel='rbf')
    elif classifier_type == 'rf':
        return cuml.ensemble.RandomForestClassifier()
    else:
        raise ValueError(f'Unsupported classifier type: {classifier_type}')


# Tune model
import optuna
import numpy as np
from sklearn.metrics import get_scorer

def objective(trial, X, y, model, cv, optimize, tune_grid):
    """
    Model is using the GPUClassifierPipeline
    """
    params = {}
    for param_name, param_grid in tune_grid.items():
        if isinstance(param_grid, tuple) and len(param_grid) == 2:
            if isinstance(param_grid[0], float):
                params[param_name] = trial.suggest_float(param_name, param_grid[0], param_grid[1])
            else:
                params[param_name] = trial.suggest_int(param_name, param_grid[0], param_grid[1])
        elif isinstance(param_grid, list):
            params[param_name] = trial.suggest_categorical(param_name, param_grid)

    model.set_params(**params)
    scores = []
    scorer = get_scorer(optimize.lower())

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # convert to array for scoring
        """" Issue with using cudf -> sklearn.metric
        print(y_train)
        print(type(y_train))
        y_true = cp.array(y_test.as_gpu_matrix())
        y_pred_arr = cp.array(y_pred.as_gpu_matrix())

        if optimize == 'f1' and y.nunique() > 2:
            score = scorer(y_true, y_pred_arr, average='macro')
        else:
            score = scorer(y_true, y_pred_arr)
        """
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    return np.mean(scores)

def opt_tune_model(X, y, cv, model, optimize, tune_grid, n_trials):
    """
    Function to tune a model using Optuna Bayesian search.

    Args:
        X: The predictors.
        y: The target label.
        cv: KFold or StratifiedKFold object for cross-validation.
        model: GPUClassifierPipeline object, the classifier to be tuned.
        optimize: The target metric to optimize towards.
        tune_grid: The hyperparameters to tune for the classifier.

    Returns:
        The best model after tuning and the score
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, model, cv, optimize, tune_grid), n_trials=n_trials)

    best_params = study.best_trial.params
    score = study.best_trial.value

    model.set_params(**best_params)

    return model, score