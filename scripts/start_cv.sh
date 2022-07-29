#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=$3
#SBATCH --partition=clara-job
#SBATCH --time=10-00:00:00
#SBATCH --mem=72G
##SBATCH --gres=gpu:v100:2
#SBATCH --gres=gpu:rtx2080ti:5
#SBATCH --output=_CV_$2_%j.log

module --ignore-cache load "CUDA/10.1.243-GCC-8.3.0"

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.4.0-fosscuda-2019b-Python-3.7.4
module load SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
module load scikit-learn/0.23.1-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4


python /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/cross_validation.py --experiment_folder=$1 --cv_name=$2 --abbreviation=$3 --config_index=$4 --test_path=$5

exit 0
EOT
