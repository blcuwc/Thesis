#!/bin/bash

TABLE_FILE1="svm_voting_output/tables.txt"
TABLE_FILE2="crf_voting_output/tables.txt"
if [ -f $TABLE_FILE1 ]; then
#  rm $TABLE_FILE1
  echo 
fi

if [ -f $TABLE_FILE2 ]; then
#  rm $TABLE_FILE2
  echo
fi

for i in 0 1 2 3 4
do 
  #python depth_analysis.py $i >> svm_voting_output/tables.txt
  python depth_analysis.py $i >> crf_voting_output/tables.txt
done
