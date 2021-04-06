#coding=utf-8

import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == "__main__":
    fold_num = sys.argv[1]
    first_fp = sys.argv[2]
    first_fn = sys.argv[3]
    second_fp = sys.argv[4]
    second_fn = sys.argv[5]
    label_name = sys.argv[6]
    first_model_name = sys.argv[7]
    second_model_name = sys.argv[8]

    first_fp_file = open(first_fp, 'r')
    first_fn_file = open(first_fn, 'r')
    second_fp_file = open(second_fp, 'r')
    second_fn_file = open(second_fn, 'r')

    first_fp_lines = first_fp_file.readlines()
    first_fn_lines = first_fn_file.readlines()
    first_fp_file.close()
    first_fn_file.close()

    second_fp_lines = second_fp_file.readlines()
    second_fn_lines = second_fn_file.readlines()
    second_fp_file.close()
    second_fn_file.close()

    first_fp_ids = []
    for line in first_fp_lines:
        #print (line)
        chunks = line.split('\t')
        first_fp_ids.append(chunks[1])

    first_fn_ids = []
    for line in first_fn_lines:
        #print (line)
        chunks = line.split('\t')
        first_fn_ids.append(chunks[1])

    if os.path.exists('error_cases/%s' % label_name):
        pass
    else:
        os.mkdir('error_cases/%s' % label_name)

    common_fp_errors = open('error_cases/%s/common_fp_errors_fold%s' % (label_name, fold_num), 'w+')
    second_uniq_fp_errors = open('error_cases/%s/%s_uniq_fp_errors%s' % (label_name, second_model_name, fold_num), 'w+')
    first_uniq_fp_errors = open('error_cases/%s/%s_uniq_fp_errors%s' % (label_name, first_model_name, fold_num), 'w+')

    common_fp_ids = []
    for line in second_fp_lines:
        #print (line)
        chunks = line.split('\t')
        if chunks[1] in first_fp_ids:
            common_fp_ids.append(chunks[1])
            new_line = chunks[1] + "\t" + chunks[2] + "\t" + chunks[3] + "\t" + chunks[4]
            #print (new_line)
            common_fp_errors.write(new_line)
        else:
            second_uniq_fp_errors.write(line)
    for line in first_fp_lines:
        #print (line)
        chunks = line.split('\t')
        if chunks[1] not in common_fp_ids:
            #print (chunks[1])
            first_uniq_fp_errors.write(line)

    common_fn_errors = open('error_cases/%s/common_fn_errors_fold%s' % (label_name, fold_num), 'w+')
    second_uniq_fn_errors = open('error_cases/%s/%s_uniq_fn_errors%s' % (label_name, second_model_name, fold_num), 'w+')
    first_uniq_fn_errors = open('error_cases/%s/%s_uniq_fn_errors%s' % (label_name, first_model_name, fold_num), 'w+')

    common_fn_ids = []
    for line in second_fn_lines:
        #print (line)
        chunks = line.split('\t')
        if chunks[1] in first_fn_ids:
            common_fn_ids.append(chunks[1])
            new_line = chunks[1] + "\t" + chunks[2] + "\t" + chunks[3] + "\t" + chunks[4]
            common_fn_errors.write(new_line)
        else:
            second_uniq_fn_errors.write(line)
    for line in first_fn_lines:
        if line.split('\t')[1] not in common_fn_ids:
            first_uniq_fn_errors.write(line)
