#!/bin/bash
#SBATCH --job-name=mpiseno-train-cliport
#SBATCH --account=iliad
#SBATCH --partition=iliad


### e.g. request 1 node with 4 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
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


# CLIPORT_TAG=roboclip_GT
# srun --output ${SLURM_LOG_DIR}/${CLIPORT_TAG}.out \
#     python nat_policies/train.py train.task=put-block-in-bowl-seen-colors \
#         tag=${CLIPORT_TAG} \
#         train.agent=roboclip \
#         train.use_gt_goals=True \
#         train.attn_stream_fusion_type=add \
#         train.trans_stream_fusion_type=conv \
#         train.goal_fusion_type=mult_ \
#         train.n_demos=1000 \
#         train.n_steps=201000 \
#         train.exp_folder=exps \
#         train.log=True \
#         dataset.cache=True \
#         train.roboclip_ckpt_path=finetune/put-block-in-bowl-seen-colors-roboclip_RN50_FiLM/checkpoints/best-v1.ckpt



CLIPORT_TAG=cliport_visual
python nat_policies/train.py train.task=put-block-in-bowl-seen-colors \
    tag=${CLIPORT_TAG} \
    train.agent=cliport_visual \
    train.attn_stream_fusion_type=add \
    train.trans_stream_fusion_type=conv \
    train.goal_fusion_type=mult_ \
    train.n_demos=1000 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    train.log=True \
    dataset.cache=True \
        










