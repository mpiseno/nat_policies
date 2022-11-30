#!/bin/bash
#SBATCH --job-name=mpiseno-train-cliport
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=20-00:00:00
#SBATCH --output=/iliad/u/mpiseno/src/nat_policies/logs/slurm_logs/train_cliport.out

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
srun python nat_policies/train.py train.task=put-block-in-bowl-seen-colors \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=1000 \
                        train.n_steps=201000 \
                        train.exp_folder=exps \
                        train.log=True \
                        dataset.cache=True










