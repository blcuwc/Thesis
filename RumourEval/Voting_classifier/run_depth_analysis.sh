#!/bin/bash

TABLE_FILE="voting_output/tables.txt"
if [ -f $TABLE_FILE]; then
  rm $TABLE_FILE
fi

for i in 0 1 2 3 4
do 
  python depth_analysis.py $i >> voting_output/tables.txt
done
