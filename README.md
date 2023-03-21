# Master thesis: Examining the suitability of Population Fidelity measures for classification problems
## Project structure

master-thesis-vt23/  
|- bin/			   # contains main files & scripts  
|- config/		   # config files, including 'experiment_settings.yml'   
|- data/		   # data for the project  
|- notebooks/		   # notebooks for EDA and implementation  
|- pickles/		   # data-objects such as models and instances of classes  
|- src/			   # source code  
|- tests/		   # test files (should mirror the src folder)  
|- Makefile		   # automization of tasks, is buggy on windows   
|- requirements.txt	   # project dependencies  
|- config_experiment.yaml  # global settings for the experiment  


## Pre-requesites
- Anaconda3/miniconda3

## Install
Run following commands  
```
conda env create --name master --file requirements.yaml
conda activate master
python -m ipykernel install --user --name master --display-name "Python 3.9.16 (master)"
```


### Experiment settings

The file `./config/experiment_settings.yml` contains all the global settings for all runs of the experiment,
such as the folders for each files, or for choosing how many synthetic datasets to generat, or how large each of 
them should be.
