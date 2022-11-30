#!/bin/bash
#SBATCH --job-name=mpiseno-finetune-clip
#SBATCH --account=iliad
#SBATCH --partition=iliad

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

CONDA_PATH=/iliad/group/cluster-support/anaconda/main/anaconda3/bin/activate
REPO_PATH=/iliad/u/mpiseno/src/nat_policies
SLURM_LOG_DIR=${REPO_PATH}/logs/slurm_logs

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID

# Setup
source ~/.bashrc
source $CONDA_PATH
unset DISPLAY

cd $REPO_PATH
conda activate michael_nat_policies

LP_TAG=finetune_clip_ViT-LP_with-vis
FT_TAG=finetune_clip_ViT-FT_with-vis
CLIP_VARIANT=ViT

# LP phase
srun --output ${SLURM_LOG_DIR}/${LP_TAG}.out \
    python nat_policies/train_latent_dynamics.py train.task=put-block-in-bowl-seen-colors \
                        tag=${LP_TAG} \
                        train.LP_phase=True \
                        train.clip_variant=${CLIP_VARIANT} \
                        train.log=True \
                        train.n_demos=1000 \
                        train.n_finetune_epochs=101 \
                        dataset.cache=True \

# FT phase
# srun --output ${SLURM_LOG_DIR}/${FT_TAG}.out \
#     python nat_policies/train_latent_dynamics.py train.task=put-block-in-bowl-seen-colors \
#                         tag=${FT_TAG} \
#                         train.LP_phase=False \
#                         train.LP_tag=${LP_TAG} \
#                         train.clip_variant=${CLIP_VARIANT} \
#                         train.log=True \
#                         train.n_demos=1000 \
#                         train.n_finetune_epochs=501 \
#                         dataset.cache=True \
                        
                        










