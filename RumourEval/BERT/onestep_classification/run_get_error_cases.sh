#!/bin/bash

for i in 0 1 2 3 4
do
  #python get_error_cases.py $i
  python depth_analysis.py $i
done
