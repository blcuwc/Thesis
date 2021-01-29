#!/bin/bash

for i in 0 1 2 3 4
do
  #python crf_svm_twostep_classification.py $i
  python crf_crf_twostep_classification.py $i
  wait
done
