#!/bin/bash

#first_prefix='../branchLSTM_cross_validation/error_cases/'
first_prefix='../BERT/onestep_classification/error_cases/'

second_prefix='../one_step_voting/svm_error_cases/'
#second_prefix='../one_step_voting/crf_error_cases/'

first_model='DistilBERT'
second_model='SVM'

for i in 0 1 2 3 4
do
  python get_unique_errors.py $i ${first_prefix}/comment_fp_error_fold$i ${first_prefix}/comment_fn_error_fold$i ${second_prefix}/comment_fp_error_fold$i ${second_prefix}/comment_fn_error_fold$i comment ${first_model} ${second_model}
  wait
  python get_unique_errors.py $i ${first_prefix}/support_fp_error_fold$i ${first_prefix}/support_fn_error_fold$i ${second_prefix}/support_fp_error_fold$i ${second_prefix}/support_fn_error_fold$i support ${first_model} ${second_model}
  wait
  python get_unique_errors.py $i ${first_prefix}/deny_fp_error_fold$i ${first_prefix}/deny_fn_error_fold$i ${second_prefix}/deny_fp_error_fold$i ${second_prefix}/deny_fn_error_fold$i deny ${first_model} ${second_model}
  wait
  python get_unique_errors.py $i ${first_prefix}/query_fp_error_fold$i ${first_prefix}/query_fn_error_fold$i ${second_prefix}/query_fp_error_fold$i ${second_prefix}/query_fn_error_fold$i query ${first_model} ${second_model}
  wait
done
