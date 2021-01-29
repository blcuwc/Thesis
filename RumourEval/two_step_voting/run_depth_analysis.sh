#!/bin/bash

#TABLE_FILE1="crf_svm_voting_output/tables.txt"
TABLE_FILE2="crf_crf_voting_output/tables.txt"

#if [ -f $TABLE_FILE1 ]; then
#  rm $TABLE_FILE1
#fi

if [ -f $TABLE_FILE2 ]; then
  rm $TABLE_FILE2
fi

for i in 0 1 2 3 4
do 
  #python depth_analysis.py $i >> crf_svm_voting_output/tables.txt
  #wait
  python depth_analysis.py $i >> crf_crf_voting_output/tables.txt
  wait
done
