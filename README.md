# Master thesis: Measuring the Utility of Synthetic Data
### An Empirical Evaluation of Population Fidelity Measures as Indicators of Synthetic Data Utility in Classification Tasks
Alexander Florean

## Project structure
./  
|- data/		   # data for the project   
|- logs/		   # contains the log file for system logging  
|- notebooks/	   # notebooks for the experiments and implementation  
|- pickles/		   # saved python-objects such as SDG models  
|- src/			   # source code, for help functions and computing PF    
|- tests/		   # test function  
|- Makefile		   # automization of tasks, has some issues in windows  
|- README.md	   # documentation file  
|- config_experiment.yaml  # global settings for the experiment    
|- requirements.yaml	   # all the packages used in the project  


## Pre-requesites
- Anaconda3/miniconda3

### Install
Run `make install` to create the environment and install all required packages.  
However, the Makefile is somewhat unpredictable in windows, have yet time to fix this yet.  

If using windows use the alternative,  
run following commands in the shell:    

```
conda env create --name master --file requirements.yaml
conda activate master
python -m ipykernel install --user --name master --display-name "Python 3.9.16 (master)"
```


### Experiment settings
The file `./config/experiment_settings.yml` contains all the global settings for all runs of the experiment,  
such as the folders for each files, or for choosing how many synthetic datasets to generat, or how large each of   
them should be.