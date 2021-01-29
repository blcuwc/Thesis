#!/bin/bash

for i in 0 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=0 /usr/local/envs/BranchLSTM/bin/python outer_get_probability.py $i
  wait
done

