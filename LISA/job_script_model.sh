#!/bin/bash
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 1:00:00

module load 2019
module load Python/3.6.6-intel-2019b
# num_trial bottleneck save_path n_gens n_agents max_model_size n_epochs
source ../venv/bin/activate

python ../iteration.py \
--num_trial $1 \
--bottleneck $2 \
--save_path ../Archive/ \
--n_generations 300 \
--n_agents 10 \
--max_model_size 10 \
--num_epochs $3 \
--shuffle_input False \
--optimizer sgdmomentum

deactivate
