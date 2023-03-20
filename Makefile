CONDA_ENV_NAME = master
PYTHON_VERSION = $(shell python --version)

#Folders
MLFLOW_LOG_DIR = notebooks/mlruns
#Folders
SYS_LOG_DIR = logs

#Filenames
REQ_FILENAME = requirements.yaml

# Todo: Fix on windows
actv:
	@conda activate $(CONDA_ENV_NAME)

# Todo: Fix on windows
clean-logs:
	@rm -r ./$(MLFLOW_LOG_DIR)/*
	@rm -r ./$(SYS_LOG_DIR)/*

save-env:
	@conda env export > ./$(REQ_FILENAME)

# Todo: Fix on windows
update-env-file:
	@conda env update  --file ./$(REQ_FILENAME) --prune


install:
	# Create the environment with necessary dependencies for the experiment
	@conda create --name $(CONDA_ENV_NAME) --file ./$(REQ_FILENAME)
	# create juptyer kernel for the current environment
	python -m ipykernel install --user --name $(CONDA_ENV_NAME) --display-name "$(PYTHON_VERSION) ($(CONDA_ENV_NAME))"

# TODO: fix on windows
test:
	$(shell pytest)

