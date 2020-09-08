#!/bin/bash

for i in 0 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=1 /data/s2230496/anaconda3/envs/ktrain/bin/python Classify.py $i
  wait
  /data/s2230496/anaconda3/envs/BranchLSTM/bin/python depth_analysis.py $i >> output/tables.txt
  wait
done

