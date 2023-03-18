import os
import cloudpickle
import yaml

import pandas as pd
from pycaret.classification import setup

def getPicklesFromDir(path: str):
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

def getExperimentConfig():
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
