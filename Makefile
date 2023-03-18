CONDA_ENV_NAME = master

#Folders
MLFLOW_LOG_DIR = notebooks/mlruns
SYS_LOG_DIR = logs
ENV_DIR = env

#Filenames
ENV_FILENAME = environment.yaml

# Todo: doesn't work yet on windows
activate-env:
	conda activate $(CONDA_ENV_NAME)

# Todo: doesn't work yet on windows
clean-logs:
	rm -r ./$(MLFLOW_LOG_DIR)/*
	rm -r ./$(SYS_LOG_DIR)/*

save-env:
	conda env export > ./$(ENV_DIR)/$(ENV_FILENAME)

# Todo: doesn't work yet on windows
update-env-file:
	conda env update  --file ./$(ENV_DIR)/$(ENV_FILENAME) --prune


# Create the environment with necessary dependencies for the experiment
INSTALL:
	conda create --name $(CONDA_ENV_NAME) --file ./$(ENV_DIR)/$(ENV_FILENAME)
