#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --exclude=clara11,clara15,clara19,clara20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=$4_$3
#SBATCH --partition=clara
#SBATCH --time=40:00:00
#SBATCH --mem=16G
##SBATCH --gres=gpu:v100:2
#SBATCH --gres=gpu:rtx2080ti:2
#SBATCH --output=$6/_ExperimentStep_$3_config_index_$4_%j.log

#module --ignore-cache load "CUDA/10.1.243-GCC-8.3.0"

#module load Python/3.7.4-GCCcore-8.3.0
#module load TensorFlow/2.4.0-fosscuda-2019b-Python-3.7.4
#module load SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
#module load scikit-learn/0.23.1-fosscuda-2019b-Python-3.7.4
#module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
#module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
#module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4

#module load Python
#module load cuDNN
#source /home/sc.uni-leipzig.de/mi186veva/venv/tf/bin/activate

module purge

module load Anaconda3/2021.11
module load cuDNN/8.9.2.26-CUDA-12.2.0

## enter the name from your env
CONDA_ENV="hsi_env"

## source your .bashrc to load conda
source ~/.bashrc

## activate conda env
conda activate hsi_env


module list

python /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/scripts/check.py

echo "Job is running on the following node(s): $SLURM_NODELIST"

python /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/cross_validator_experiment.py --experiment_folder=$1 --cv_name=$2 --abbreviation=$3 --config_index=$4 --results_folder=$5

conda deactivate
exit 0
EOT
