#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --exclude=clara11,clara15,clara19,clara20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=$6
#SBATCH --partition=clara
#SBATCH --time=40:00:00
#SBATCH --mem=32G
##SBATCH --gres=gpu:v100:2
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --output=$4/$5/_$6_%j.log


echo $1
echo $2
echo $3
echo $4
echo $5
echo $6

module purge

module load Anaconda3/2021.11
module load cuDNN/8.9.2.26-CUDA-12.2.0

CONDA_ENV="hsi_env"

source ~/.bashrc

conda activate hsi_env

module list

python check.py

echo "Job is running on the following node(s): $SLURM_NODELIST"

python /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/cross_validators/cross_validation_parallel_steps.py --model_name=$1 --all_patients=$2 --leave_out_names=$3

conda deactivate

exit 0
EOT
