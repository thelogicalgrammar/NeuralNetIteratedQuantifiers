#!/bin/bash
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 10:00:00

module load 2019
module load Python/3.6.6-intel-2019b
# num_trial bottleneck save_path n_gens n_agents max_model_size n_epochs
source ../venv/bin/activate
for i in {1..20}
do
	echo($i) & \
	python ../iteration.py \
	--num_trial $i \
	--bottleneck $1 \
	--save_path ./Archive \
	--n_generations 300 \
	--n_agents 10 \
	--max_model_size 10 \
	--n_epochs $2 \
	--shuffle_input False \
	--optimizer sgd &
done
deactivate
