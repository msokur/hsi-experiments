# HSI-Experiments

## Installation Guide
This instruction are for using a terminal. You can also do the next steps with your IDE.

### 1. Clone the repository
Open a terminal and navigate to the directory where the repository should be saved.
Then clone the repository.
```
git clone https://git.iccas.de/MaktabiM/hsi-experiments
```
### 2. Create conda environment and install requirements
Navigate into the repository folder with the terminal
```
cd hsi-experiments
```
Create a conda env with the python version `3.10.4`, the env name `hsi_env` and activate the environment.
```
conda create --name hsi_env python=3.10.4
conda activate hsi_env
```
When the env ist activ you see the name of this env before your current path in the terminal.

Now you can install the needed requirements form the given .txt files. You can install once the requirements to use only 
CPU or also use your GPU (only for Nvidia GPU) for training.

#### 2.1 Only CPU use
This requirement ist normally for your local machine.
```
pip install -r requirements_only_cpu.txt
```

#### 2.2 CPU and GPU use
This requirement you should use when you work with the cluster.
```
pip install -r requirements_gpu.txt
```

## 3. Configurations
To use and edit the configurations in the program see these links.<br>
https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/First%20Steps/Configuration%20Templates <br>
https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/First%20Steps/Edit%20get_config

## 4. Run a python file
At your local machine you can start the python file with the command.
```
python 'path/to/your/python/file.py'
```
You have to be sure, that the conde env `hsi_env` is activated.

For the cluster we use job files, to run our program on the different nodes.
The basic job file looks like this:
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

#SBATCH --job-name=r_cv

#SBATCH --partition=clara
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
##SBATCH --gres=gpu:v100:2
#SBATCH --gres=gpu:rtx2080ti:2

#SBATCH --output=output/output_file_%j.log

module load Anaconda3/2023.09-0

## enter the name from your env
CONDA_ENV="hsi_env"

## source your .bashrc to load conda
source ~/.bashrc

## activate conda env
conda activate $CONDA_ENV

## run python script
## edit here the path to the file you want to start
python 'path/to/your/python/file.py'

## deactivate conda env
conda deactivate
```
