#!/bin/bash

params_file="voting_output/best_parameters.txt"
weights_file="voting_output/feature_weights.txt"

if [ -f $params_file ]; then
  rm $params_file
fi

if [ -f $weights_file ]; then
  rm $weights_file
fi

for i in 0 1 2 3 4
do
  #/usr/local/envs/DistillBert/bin/python Voting_classifier.py $i
  python Voting_classifier.py $i
  wait
done
