#!/bin/bash

for ntrial in {1..20}
do
	for bneck in 200 512 715 1024 
	do
		for nepoch in 4 8
		do
			sbatch job_script_model.sh $ntrial $bneck $nepoch
		done
	done
done
