## Project structure

master-thesis-vt23/
|- bin/			# contains main files & scripts
|- config/		# config files, including 'experiment_settings.yml' 
|- data/		# data for the project
|- notebooks/		# notebooks for EDA and implementation
|- pickles/		# data-objects such as models and instances of classes
|- src/			# source code
|- tests/		# test files (should mirror the src folder)
|- Makefile		# automization of tasks
|- requirements.txt	# project dependencies

## Set-up jupyter kernel environment right
Run the following in the wanted environment:
`pip install ipykernel`
`python -m ipykernel install --user --name ENVNAME --display-name "Python (whatever you want to call it)"`


### Experiment settings

(not implemented yet, just an idea right now).

The file `./config/experiment_settings.yml` contains all the global settings for all runs of the experiment,
such as the folders for each files, or for choosing how many synthetic datasets to generat, or how large each of 
them should be.