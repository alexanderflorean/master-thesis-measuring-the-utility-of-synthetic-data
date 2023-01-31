## Project structure

master-thesis-vt23/
|- bin/			# contains main files & scripts
|- config/		# config files
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
