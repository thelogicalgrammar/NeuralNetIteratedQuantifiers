#!/bin/bash

for bneck in 200 512 715 1024 
do
	for nepoch in 4 8
	do
		sbatch job_script_model.sh $bneck $nepoch
	done
done
