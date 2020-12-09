#!/bin/bash 
#SBATCH -p  gtx1080
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc2u18@soton.ac.uk
#SBATCH --time=60:00:00

module load conda
source activate pytorch_env
echo $@
python /home/cc2u18/Weight_Fix_Networks/all_conv_experiments.py $@
