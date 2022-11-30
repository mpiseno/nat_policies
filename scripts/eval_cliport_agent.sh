#!/bin/bash
#SBATCH --job-name=mpiseno-train-cliport
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=20-00:00:00
#SBATCH --output=/iliad/u/mpiseno/src/nat_policies/logs/slurm_logs/eval_cliport.out

### only use the following if you want email notification
### SBATCH --mail-user=mpiseno@stanford.edu
### SBATCH --mail-type=ALL

### e.g. request 1 node with 4 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB

CONDA_PATH=/iliad/group/cluster-support/anaconda/main/anaconda3/bin/activate
REPO_PATH=/iliad/u/mpiseno/src/nat_policies

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID

# Setup
source ~/.bashrc
source $CONDA_PATH
unset DISPLAY

cd $REPO_PATH
conda activate michael_nat_policies

### the command to run
srun python nat_policies/eval.py eval_task=put-block-in-bowl-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps 










