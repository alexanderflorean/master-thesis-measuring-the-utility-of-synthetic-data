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
|- Makefile		   # automization of tasks
|- requirements.txt	   # project dependencies
|- config_experiment.yaml  # global settings for the experiment


## Pre-requesites
- Anaconda3/miniconda3

- For windows users:
  - make: a command linue utility that executes makefile (standard program in linux OS'es).
    - Tip, use "Chocolatey": package manager, can be used to install make 
      (![install instructions](https://chocolatey.org/install)) 
      Then run ´choco install make´ 

## Install

### Set-up jupyter kernel environment right
Run the following in the wanted environment:
`pip install ipykernel`
`python -m ipykernel install --user --name ENVNAME --display-name "Python (whatever you want to call it)"`

start R and run "install.packages("synthpop")" then "library(synthpop)"

conda create --name env_name python R:
conda install -c conda-forge rpy2 synthpop

### Experiment settings

The file `./config/experiment_settings.yml` contains all the global settings for all runs of the experiment,
such as the folders for each files, or for choosing how many synthetic datasets to generat, or how large each of 
them should be.
