#!/usr/bin/env bash

mkdir "$HOME"/output_dir
paths='$TMPDIR/Archive/*/quantifiers.npy'

for i in `seq 0 9`; 
do
    begin=$((16*$i+1))
    end=$(($begin+15))
    paths=$(sed -n ${begin},${end}p pathnames.txt)
    sbatch job_script_analysis.sh "$paths"
done
