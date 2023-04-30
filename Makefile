CONDA_ENV_NAME = master
PYTHON_VERSION = $(shell python --version)

#Folders
MLFLOW_LOG_DIR = notebooks/mlruns
#Folders
SYS_LOG_DIR = logs

#Filenames
REQ_FILENAME = requirements.yaml

actv:
	@conda activate $(CONDA_ENV_NAME)

save-env:
	@conda env export > ./$(REQ_FILENAME)

update-env-file:
	@conda env update  --file ./$(REQ_FILENAME) --prune


install:
	# Create the environment with necessary dependencies for the experiment
	@conda env create --name $(CONDA_ENV_NAME) --file $(REQ_FILENAME)
	# create juptyer kernel for the current environment
	@conda activate $(CONDA_ENV_NAME)
	@python -m ipykernel install --user --name $(CONDA_ENV_NAME) --display-name "$(PYTHON_VERSION) ($(CONDA_ENV_NAME))"

test:
	@pytest

clean:
	PowerShell -Command "Remove-Item 'C:\Users\flore\source\repos\master-thesis-vt23\logs\*' -Exclude '.gitkeep' -Recurse -Force -ErrorAction SilentlyContinue"
	PowerShell -Command "Remove-Item 'C:\Users\flore\source\repos\master-thesis-vt23\notebooks\mlruns\*' -Recurse -Force -ErrorAction SilentlyContinue"
	PowerShell -Command "Remove-Item 'C:\Users\flore\source\repos\master-thesis-vt23\data\synthetic\*.csv' -Exclude '.gitkeep' -Recurse -Force -ErrorAction SilentlyContinue"
	PowerShell -Command "Remove-Item 'C:\Users\flore\source\repos\master-thesis-vt23\data\result\model_performance.csv' -Force -ErrorAction SilentlyContinue"

