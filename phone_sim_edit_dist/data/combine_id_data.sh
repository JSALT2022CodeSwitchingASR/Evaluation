#!/usr/bin/env bash

files='A1 A2 A3 A4 hyp ref'
sed -i -r '/^\s*$/d' ids

for f in $files; do 
    # remove empty lines 
    
    echo $f
    paste ids $f  | column -s $'\t' -t > ${f}_with_ids
	#sed -i -r '/^\s*$/d' $f

done


