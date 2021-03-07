#!/bin/bash

params_file="svm_voting_output/best_parameters.txt"
weights_file="svm_voting_output/feature_weights.txt"
latex_file="svm_voting_output/feature_weights_latex.txt"

if [ -f $params_file ]; then
  rm $params_file
fi

if [ -f $weights_file ]; then
  rm $weights_file
fi

if [ -f $latex_file ]; then
  rm $latex_file
fi

for i in 0 1 2 3 4
do
  #/usr/local/envs/DistillBert/bin/python Voting_classifier.py $i
  #python SVM_Voting_classifier.py $i
  python crf_classification.py $i
  wait
done
