import os
import cloudpickle
import yaml
import time
import pandas as pd

from functools import wraps
from pycaret.classification import setup
from sdmetrics.utils import get_columns_from_metadata, get_type_from_column_meta

def getPicklesFromDir(path: str) -> list[dict]:
    """    Returns all pickles in the provided path as a list.

    In: 
        'path': the relative path to the destination directory

    Out: 
        List of all pickles in the provided path

       
    """

    pickles = [] 

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            folder = os.path.join(dirname, filename)
            pickle_obj = cloudpickle.load(open(folder, "rb"))
            pickles.append(pickle_obj)

    return pickles

def getExperimentConfig() -> dict:
    """     Returns the YAML experiment configuration that contains all 
    global experiment settings

    """

    with open('../experiment_config.yml', 'r') as file:
        config = yaml.safe_load(file)

    return config

def run_pycaret_setup(data_path: str, setup_param: dict):
    """     A wrapper function for the experiment to run pycaret setup()

            sends the correct params to the pycaret setup() function and 
            returns its return value.
            Thus enabling iterative runs of the settings.
    """
    pycaret_setup = setup(
        data = pd.read_csv(data_path),
        **setup_param 
    )
    return pycaret_setup

def get_categorical_indicies(data:pd.DataFrame, metadata:dict) -> list[int]:
    """ Returns a list of indices of the categorical columns in the dataset """
    
    indices = []
    
    columns = get_columns_from_metadata(metadata)
    for col in columns:
        col_type = get_type_from_column_meta(columns[col])
        if col_type == 'categorical' or col_type == 'boolean':
            col_index = data.columns.get_loc(col)
            indices.append(col_index)
            
    return indices

def timefunction(func):
    @wraps(func)
    def timefunction_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        return result, total_time
    return timefunction_wrapper