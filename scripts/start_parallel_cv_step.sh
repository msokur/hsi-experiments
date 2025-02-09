#!/bin/bash
sbatch <<EOT
#!/bin/bash
##SBATCH --exclude=clara11,clara15,clara19,clara20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=$2
#SBATCH --partition=clara
#SBATCH --time=40:00:00
#SBATCH --mem=32G
##SBATCH --gres=gpu:v100:2
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --output=$1/_$2_%j.log


echo $1
echo $2
#echo $3

## remove all previously loaded modules
module purge

## source your .bashrc to load conda
source ~/.bashrc

## deactivate the conda base env
conda deactivate

## load the anaconda module
module load Anaconda3/2023.09-0

## enter the name from your env
CONDA_ENV="hsi"
## CONDA_ENV="hsi_env"
## CONDA_ENV="tf"

## activate conda env
conda activate $CONDA_ENV

## load repository variables
source repo_variables.sh

## load the check.py
python /home/sc.uni-leipzig.de/ze43enib/Peritoneum/hsi-experiments/scripts/check.py
echo "Job is running on the following node(s): $SLURM_NODELIST"

python /home/sc.uni-leipzig.de/ze43enib/Peritoneum/hsi-experiments/cross_validators/parallel_CV_step.py --CV_folder=$1 --CV_step_name=$2
#--all_patients=$2 --leave_out_names=$3

conda deactivate

exit 0
EOT
