#!/bin/bash

#SBATCH --job-name=mpiseno-cliport-data-collection
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

CONDA_PATH=/iliad/group/cluster-support/anaconda/main/anaconda3/bin/activate
REPO_PATH=/iliad/u/mpiseno/src/nat_policies
CLIPORT_PATH=/iliad/u/mpiseno/src/cliport
SLURM_LOG_DIR=${REPO_PATH}/logs/slurm_logs

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID

# Setup
source ~/.bashrc
source $CONDA_PATH
#unset DISPLAY

cd $CLIPORT_PATH
conda activate michael_nat_policies


TASK=packing-boxes-pairs-seen-colors
SLURM_TAG=cliport-data-collection_${TASK}

### the command to run
srun --ntasks=1 --output ${SLURM_LOG_DIR}/${SLURM_TAG}.out \
    python cliport/demos.py n=100 \
        task=${TASK} \
        mode=val 

srun --ntasks=1 --output ${SLURM_LOG_DIR}/${SLURM_TAG}.out \
    python cliport/demos.py n=100 \
        task=${TASK} \
        mode=test

#srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "echo 'hello 1'"
#srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "echo 'hello 2'"
        










