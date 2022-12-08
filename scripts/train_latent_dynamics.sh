#!/bin/bash
#SBATCH --job-name=mpiseno-finetune-clip
#SBATCH --account=iliad
#SBATCH --partition=iliad

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

# Set your conda path
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
conda activate michael_nat_policies # Set your conda environment

# Variables for config (see nat_policies/cfg/finetune_clip.yaml)
CLIP_VARIANT=ViT
LP_TAG=finetune_clip_ViT_fusion-FiLM_LP
FT_TAG=finetune_clip_ViT_fusion-FiLM_FT


# LP phase - for concat and FiLM fusion
# srun --output ${SLURM_LOG_DIR}/${LP_TAG}.out \
#     python nat_policies/train_latent_dynamics.py train.task=put-block-in-bowl-seen-colors \
#                         tag=${LP_TAG} \
#                         train.fusion_type=FiLM \
#                         train.LP_phase=True \
#                         train.clip_variant=${CLIP_VARIANT} \
#                         train.log=True \
#                         train.n_demos=1000 \
#                         train.n_finetune_epochs=101 \
#                         dataset.cache=True \

# FT phase - for concat and FiLM fusion
srun --output ${SLURM_LOG_DIR}/${FT_TAG}.out \
    python nat_policies/train_latent_dynamics.py train.task=put-block-in-bowl-seen-colors \
                        tag=${FT_TAG} \
                        train.fusion_type=FiLM \
                        train.LP_phase=False \
                        train.LP_tag=${LP_TAG} \
                        train.clip_variant=${CLIP_VARIANT} \
                        train.log=True \
                        train.n_demos=1000 \
                        train.n_finetune_epochs=501 \
                        dataset.cache=True \
                        
                        










