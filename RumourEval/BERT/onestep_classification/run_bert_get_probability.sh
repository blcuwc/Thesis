#!/bin/bash

for i in 0 1 2 3 4
do
  probability_path="./saved_probability/fold$i/"
  if [ -d $probability_path ]; then
    rm -rf $probability_path
  fi

  CUDA_VISIBLE_DEVICES=0 /usr/local/envs/DistillBert/bin/python Classify_get_probability.py $i
  wait
done

