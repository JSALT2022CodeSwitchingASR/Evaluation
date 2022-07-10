#!/usr/bin/env bash

dir=$1
target_files='A1 A2 A3 A4 ref'
source_file='hyp'

for target in $target_files; do 
    
    echo $f
	python psd.py $dir/${source_file}_with_ids $dir/${target}_with_ids
    

done