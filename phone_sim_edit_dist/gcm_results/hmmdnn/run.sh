#!/usr/bin/env bash


#pref='emad'
target_files="exp0 exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9 exp10"
source_file=test

for target in $target_files; do 
    
	echo "escwa_${target} escwa_${source_file}"
	python psd2.py escwa_${target} escwa_${source_file}
    

done