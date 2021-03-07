# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
import pandas as pd

table_four_head = r"Depth"
table_five_head = r"Lab"
table_three_head = r"Accuracy"
class_labels = ["Support", "Deny", "Query", "Comment"]

in_table_four = False
in_table_five = False
in_table_three = False

table_four_info = {}
table_five_info = {}
table_three_info = {}

def Get_average_table_four(table_four_info):
    depth = ["0", "1", "2", "3", "4", "5", "6+"]
    sum_table_four = {}
    for i in depth:
        sum_table_four[i] = []
    for fold_num, table in table_four_info.items():
        for depth, depth_info in table.items():
            temp_list = []
            for i in range(len(depth_info)):
                if i < 5:
                    temp_list.append(int(depth_info[i]))
                else:
                    temp_list.append(float(depth_info[i]))
            if sum_table_four[depth] == []:
                sum_table_four[depth] = temp_list
            else:
                for i in range(len(temp_list)):
                    sum_table_four[depth][i] += temp_list[i]
                if fold_num == 4:
                    for i in range(len(sum_table_four[depth])):
                        if i > 4:
                            sum_table_four[depth][i] = sum_table_four[depth][i] / 5
        #table_four_info[fold_num] = table
    #print (table_four_info)
    #print (sum_table_four)

    table_DF = pd.DataFrame.from_dict(sum_table_four, orient='index')
    table_DF = table_DF.reset_index()
    table_DF.columns = ["Depth", "# tweets", "# Support", "# Deny", "# Query", "# Comment", "Accuracy", "MacroF","Support", "Deny", "Query", "Comment"]
    #print (table_DF)
    print (table_DF.to_latex(index=False))

def Get_average_table_five(table_five_info):
    sum_confusion_matrix = np.zeros((4,4))
    for fold_num, confusion_matrix in table_five_info.items():
        sum_confusion_matrix += confusion_matrix
    #print (sum_confusion_matrix)
    DF = pd.DataFrame(sum_confusion_matrix, columns = ["Comment", "Deny", "Query", "Support"], index = ["Comment", "Deny", "Query", "Support"], dtype = np.int64)
    #print (DF)
    print (DF.to_latex(index=False))

def Get_average_table_three(table_three_info):
    sum_table_three = []
    for fold_num, fold_list in table_three_info.items():
        temp_list = []
        for i in fold_list:
            temp_list.append(float(i))

        if sum_table_three == []:
            sum_table_three = temp_list
        else:
            sum_table_three = [sum_table_three[i] + temp_list[i] for i in range(len(temp_list))]

        if fold_num == 4:
            average_table_three = [sum_table_three[i] / 5 for i in range(len(sum_table_three))]
    #print (average_table_three)
    df = pd.DataFrame(np.array([average_table_three]), dtype=np.float64, columns = ["Accuracy", "Precision", "Recall", "F-score"])
    print (df)

if __name__ == "__main__":

    # Read result file
    result_file = sys.argv[1]

    with open(result_file) as f:
        fold_num = 0
        table_four_fold = {}
        table_three_fold = []

        confusion_matrix = np.zeros((4,4))
        confusion_matrix_layer = 0

        for line in f:
            if in_table_four:
                depth = line.strip().split()[0]
                depth_info = line.strip().split()[1:]
                table_four_fold[depth] = depth_info
                if depth == "6+":
                    table_four_info[fold_num] = table_four_fold
                    table_four_fold = {}
                    #fold_num += 1
                    in_table_four = False
                continue

            if in_table_five:
                confusion_matrix[confusion_matrix_layer] = line.strip().split()[1:]
                confusion_matrix_layer += 1
                if confusion_matrix_layer == 4:
                    table_five_info[fold_num] = confusion_matrix
                    confusion_matrix = np.zeros((4,4))
                    confusion_matrix_layer = 0
                    in_table_five = False
                continue

            if in_table_three:
                if re.match(r"Precision", line):
                    precision = line.strip().split()[1]
                    table_three_fold.append(precision)
                    continue
                if re.match(r"Recall", line):
                    recall = line.strip().split()[1]
                    table_three_fold.append(recall)
                    continue
                if re.match(r"F-score", line):
                    fscore = line.strip().split()[1]
                    table_three_fold.append(fscore)
                    table_three_info[fold_num] = table_three_fold
                    table_three_fold = []
                    fold_num += 1
                    in_table_three = False
                continue

            if re.match(table_four_head, line):
                in_table_four = True
                continue

            if re.match(table_five_head, line):
                in_table_five = True
                continue
            
            if re.match(table_three_head, line):
                accuracy = line.strip().split(" = ")[1]
                table_three_fold.append(accuracy)
                in_table_three = True
                continue
    #print (table_four_info)
    #print (table_five_info)
    #print (table_three_info)
    Get_average_table_four(table_four_info)
    Get_average_table_five(table_five_info)
    Get_average_table_three(table_three_info)
