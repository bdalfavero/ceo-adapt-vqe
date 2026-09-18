#!/bin/bash --login

#SBATCH --array=0-4
#SBATCH --time=48:00:00
#SBATCH --mem=25G
#SBATCH --job-name=xxz_max_chi
#SBATCH --mail-user=dalfaver@msu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -e

n_vals=(40 50 60 80 100)
N=${n_vals[$SLURM_ARRAY_TASK_ID]}
chi=1000
# echo $SLURM_ARRAY_TASK_ID $N $chi

module purge
module load Miniforge3
conda activate adapt

python ~/ceo-adapt-vqe/examples/Scaling/xxz_tn_batch.py $N $chi --num-iter=300

conda deactivate
