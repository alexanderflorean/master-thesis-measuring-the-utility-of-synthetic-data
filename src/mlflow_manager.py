import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from typing import Optional
import pandas as pd
import os


class MLFlowManager:
    """
    The MLFlowManager class is a high-level interface for managing MLflow experiments, runs, logging, and artifacts.
    """

    def __init__(self, experiment_name: str):
        """
        Initialize a new MLFlowManager for the given experiment name.

        Parameters:
        ------------
        experiment_name: str
            The name of the MLflow experiment.
        """
        self.experiment_name = experiment_name
        self.mlflow_client = MlflowClient()
        self.experiment_id = self.mlflow_client.create_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new run in the current experiment.

        Parameters:
        ------------
        run_name: Optional[str]
            The name of the run. Optional.

        Return:
        -------
        run: mlflow.entities.Run
            The newly created run.
        """
        self.run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        return self.run

    def log_params(self, params: dict):
        """
        Log a set of parameters as key-value pairs.

        Parameters:
        ------------
        params: dict
            A dictionary containing the key-value pairs to be logged.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict):
        """
        Log a set of parameters as key-value pairs.

        Parameters:
        ------------
        params: dict
            A dictionary containing the key-value pairs to be logged.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_metric(self, key: str, value: float):
        """
        Log a single key-value pair metric for the current run.

        Parameters:
        ------------
        key: str
            The key of the metric.
        value: float
            The value of the metric.
        """
        mlflow.log_metric(key, value)

    def save_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Save a local file or directory as an artifact of the current run.

        Parameters:
        ------------
        local_path: str
            The local path of the file or directory to be saved as an artifact.
        artifact_path: Optional[str]
            The destination path within the run's artifact URI. Optional.
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None, **kwargs):
        """
        Log a scikit-learn model as an artifact of the current run.

        Parameters:
        ------------
        model: sklearn model
            The scikit-learn model to be logged.
        artifact_path: str
            The destination path within the run's artifact URI.
        registered_model_name: Optional[str]
            The name of the registered model. Optional.
        **kwargs:
            Additional keyword arguments passed to `mlflow.sklearn.log_model()`.
        """
        mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)

    def set_synthetic_data_tag(self, is_synthetic: bool):
        """
        Set a tag indicating if the model was trained on a synthetic or original dataset.

        Parameters:
        ------------
        is_synthetic: bool
            True if the model was trained on a synthetic dataset, False otherwise.
        """
        tag_value = "synthetic" if is_synthetic else "original"
        mlflow.set_tag("dataset_type", tag_value)

    def save_test_holdout_data(self, test_data: pd.DataFrame, artifact_path: str = "test_holdout_data"):
        """
        Save the test-holdout data as an artifact of the current run.
    Parameters:
    ------------
    test_data: pd.DataFrame
        The test-holdout data as a pandas DataFrame.
    artifact_path: str
        The destination path within the run's artifact URI. Optional.
    """
    test_data.to_csv("test_holdout_data.csv", index=False)
    self.save_artifact("test_holdout_data.csv", artifact_path)

    def end_run(self):
        """
        End the current run.
        """
        mlflow.end_run()

    def load_run_by_name(self, run_name: str):
        """
        Load a run by its name.

        Parameters:
        ------------
        run_name: str
            The name of the run.

        Return:
        -------
        run: mlflow.entities.Run
            The found run.

        Raise:
        ------
        ValueError:
            If no run found with the given name.
        """
        runs = self.mlflow_client.search_runs(self.experiment_id, f"tag.mlflow.runName='{run_name}'")
        if runs:
            return runs[0]
        else:
            raise ValueError(f"No run found with the name '{run_name}'")

    def get_run_by_id(self, run_id: str):
        """
        Get a run by its ID.

        Parameters:
        ------------
        run_id: str
            The ID of the run.

        Return:
        -------
        run: mlflow.entities.Run
            The found run.
        """
        return self.mlflow_client.get_run(run_id)

    def get_test_holdout_data(self, run_id: str, artifact_path: str = "test_holdout_data") -> pd.DataFrame:
        """
        Get the test-holdout data for a specific run by its ID.

        Parameters:
        ------------
        run_id: str
            The ID of the run.
        artifact_path: str
            The artifact path where the test-holdout data is stored. Optional.

        Return:
        -------
        test_holdout_data: pd.DataFrame
            The test-holdout data as a pandas DataFrame.

        Raise:
        ------
        ValueError:
            If test-holdout data is not found for the given run ID.
        """
        artifact_uri = self.mlflow_client.get_artifact_uri(run_id, artifact_path)
        local_path = mlflow.get_artifact_uri().replace("file://", "")
        local_file_path = os.path.join(local_path, run_id, artifact_path, "test_holdout_data.csv")
        
        if os.path.exists(local_file_path):
            return pd.read_csv(local_file_path)
        else:
            raise ValueError(f"Test-holdout data not found for run ID '{run_id}' at '{local_file_path}'")

    def get_run_id_by_name(self, run_name: str) -> Optional[str]:
        """
        Get a run ID by its name.

        Parameters:
        ------------
        run_name: str
            The name of the run.

        Return:
        -------
        run_id: Optional[str]
            The ID of the found run, or None if not found.
        """
        runs = self.mlflow_client.search_runs(self.experiment_id, f"tag.mlflow.runName='{run_name}'")
        if runs:
            return runs[0].info.run_id
        else:
            return None

