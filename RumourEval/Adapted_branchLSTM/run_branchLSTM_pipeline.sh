#!/bin/bash

for i in 0 1 2 3 4
do
  #python preprocessing.py $i
  #wait
  #python outer.py $i
  #wait
  python depth_analysis.py $i
  wait
done

