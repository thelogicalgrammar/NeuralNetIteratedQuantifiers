#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00

module load 2019
module load Python/3.6.6-intel-2019b
# num_trial bottleneck save_path n_gens n_agents max_model_size n_epochs
source ../venv/bin/activate
python ../analysis.py --mode 'convert' --file_pattern $1
deactivate
