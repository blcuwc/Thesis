#!/bin/bash

for i in 0 1 2 3 4
do
  python get_unique_errors.py $i ../branchLSTM_cross_validation/error_cases/comment_fp_error_fold$i ../branchLSTM_cross_validation/error_cases/comment_fn_error_fold$i ../BERT_baseline/onestep_classification/error_cases/comment_fp_error_fold$i ../BERT_baseline/onestep_classification/error_cases/comment_fn_error_fold$i comment
  wait
  python get_unique_errors.py $i ../branchLSTM_cross_validation/error_cases/support_fp_error_fold$i ../branchLSTM_cross_validation/error_cases/support_fn_error_fold$i ../BERT_baseline/onestep_classification/error_cases/support_fp_error_fold$i ../BERT_baseline/onestep_classification/error_cases/support_fn_error_fold$i support
  wait
  python get_unique_errors.py $i ../branchLSTM_cross_validation/error_cases/deny_fp_error_fold$i ../branchLSTM_cross_validation/error_cases/deny_fn_error_fold$i ../BERT_baseline/onestep_classification/error_cases/deny_fp_error_fold$i ../BERT_baseline/onestep_classification/error_cases/deny_fn_error_fold$i deny
  wait
  python get_unique_errors.py $i ../branchLSTM_cross_validation/error_cases/query_fp_error_fold$i ../branchLSTM_cross_validation/error_cases/query_fn_error_fold$i ../BERT_baseline/onestep_classification/error_cases/query_fp_error_fold$i ../BERT_baseline/onestep_classification/error_cases/query_fn_error_fold$i query
  wait
done
