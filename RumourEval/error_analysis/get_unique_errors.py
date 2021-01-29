#coding=utf-8

import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == "__main__":
    fold_num = sys.argv[1]
    BranchLSTM_fp = sys.argv[2]
    BranchLSTM_fn = sys.argv[3]
    DistillBert_fp = sys.argv[4]
    DistillBert_fn = sys.argv[5]
    label_name = sys.argv[6]

    Bfp_file = open(BranchLSTM_fp, 'r')
    Bfn_file = open(BranchLSTM_fn, 'r')
    Dfp_file = open(DistillBert_fp, 'r')
    Dfn_file = open(DistillBert_fn, 'r')

    Bfp_lines = Bfp_file.readlines()
    Bfn_lines = Bfn_file.readlines()
    Bfp_file.close()
    Bfn_file.close()

    Dfp_lines = Dfp_file.readlines()
    Dfn_lines = Dfn_file.readlines()
    Dfp_file.close()
    Dfn_file.close()

    Bfp_ids = []
    for line in Bfp_lines:
        #print (line)
        chunks = line.split('\t')
        Bfp_ids.append(chunks[1])

    Bfn_ids = []
    for line in Bfn_lines:
        #print (line)
        chunks = line.split('\t')
        Bfn_ids.append(chunks[1])

    if os.path.exists('error_cases/%s' % label_name):
        pass
    else:
        os.mkdir('error_cases/%s' % label_name)

    common_fp_errors = open('error_cases/%s/common_fp_errors_fold%s' % (label_name, fold_num), 'w+')
    Duniq_fp_errors = open('error_cases/%s/Duniq_fp_errors%s' % (label_name, fold_num), 'w+')
    Buniq_fp_errors = open('error_cases/%s/Buniq_fp_errors%s' % (label_name, fold_num), 'w+')

    common_fp_ids = []
    for line in Dfp_lines:
        #print (line)
        chunks = line.split('\t')
        if chunks[1] in Bfp_ids:
            common_fp_ids.append(chunks[1])
            new_line = chunks[1] + "\t" + chunks[2] + "\t" + chunks[3] + "\t" + chunks[4]
            #print (new_line)
            common_fp_errors.write(new_line)
        else:
            Duniq_fp_errors.write(line)
    for line in Bfp_lines:
        #print (line)
        chunks = line.split('\t')
        if chunks[1] not in common_fp_ids:
            #print (chunks[1])
            Buniq_fp_errors.write(line)

    common_fn_errors = open('error_cases/%s/common_fn_errors_fold%s' % (label_name, fold_num), 'w+')
    Duniq_fn_errors = open('error_cases/%s/Duniq_fn_errors%s' % (label_name, fold_num), 'w+')
    Buniq_fn_errors = open('error_cases/%s/Buniq_fn_errors%s' % (label_name, fold_num), 'w+')

    common_fn_ids = []
    for line in Dfn_lines:
        #print (line)
        chunks = line.split('\t')
        if chunks[1] in Bfn_ids:
            common_fn_ids.append(chunks[1])
            new_line = chunks[1] + "\t" + chunks[2] + "\t" + chunks[3] + "\t" + chunks[4]
            common_fn_errors.write(new_line)
        else:
            Duniq_fn_errors.write(line)
    for line in Bfn_lines:
        if line.split('\t')[1] not in common_fn_ids:
            Buniq_fn_errors.write(line)
