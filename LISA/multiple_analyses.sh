#!/usr/bin/env bash

for path in ../Archive/*/quantifiers.npy
do
	echo $path
    sbatch job_script_analysis.sh $path
done
