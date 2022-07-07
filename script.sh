#!/bin/bash 
#SBATCH -p ecsstaff  
#SBATCH -A ecsstaff
#SBATCH --mem=100G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc2u18@soton.ac.uk
#SBATCH --time=90:00:00
#SBATCH --gres-flags=enforce-binding

module load conda
source activate pytorch
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

#export  NCCL_SOCKET_IFNAME=^docker0,lo

echo $@
python /home/cc2u18/Prev_WFN/Weight_Fix_Networks/pre_trained_model_experiments.py $@
