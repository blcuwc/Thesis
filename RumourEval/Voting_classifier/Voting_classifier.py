# -*- coding: utf-8 -*-
#environment : ktrain

import sys
import sklearn
import pickle
import os
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from preprocessing import load_dataset, Cross_validation_threads
from Classify import Extract_dataset

def preprocess_probabilities(path_branch, path_distill, dataset):
    train_labels = dataset['train'][1]
    dev_labels = dataset['dev'][1]
    test_labels = dataset['test'][1]

    train_ids = dataset['train'][2]
    dev_ids = dataset['dev'][2]
    test_ids = dataset['test'][2]

    categories = ['support', 'query', 'deny', 'comment']
    categories2num = {'support':0, 'query':1, 'deny':2, 'comment':3}

    train_id_label = { ID:categories2num[label] for ID, label in zip(train_ids, train_labels)}
    dev_id_label = { ID:categories2num[label] for ID, label in zip(dev_ids, dev_labels)}
    test_id_label = { ID:categories2num[label] for ID, label in zip(test_ids, test_labels)}

    #load branchLSTM three dataset probabilities
    branch_train_pro_file = open(os.path.join(path_branch, "train/probabilities.txt"), "rb")
    branch_train_pro_dict = pickle.load(branch_train_pro_file, encoding = "latin1")
    branch_train_id_pro = { ID:[train_id_label[ID], probabilities] for ID, probabilities in zip(branch_train_pro_dict["ID"], branch_train_pro_dict["Probabilities"])
                           if ID in train_id_label}

    branch_dev_pro_file = open(os.path.join(path_branch, "dev/probabilities.txt"), "rb")
    branch_dev_pro_dict = pickle.load(branch_dev_pro_file, encoding = "latin1")
    branch_dev_id_pro = { ID:[dev_id_label[ID], probabilities] for ID, probabilities in zip(branch_dev_pro_dict["ID"], branch_dev_pro_dict["Probabilities"])
                         if ID in dev_id_label}

    branch_test_pro_file = open(os.path.join(path_branch, "test/probabilities.txt"), "rb")
    branch_test_pro_dict = pickle.load(branch_test_pro_file, encoding = "latin1")
    branch_test_id_pro = { ID:[test_id_label[ID], probabilities] for ID, probabilities in zip(branch_test_pro_dict["ID"], branch_test_pro_dict["Probabilities"])
                          if ID in test_id_label}

    #load distilbert three dataset probabilities
    distill_train_pro_file = open(os.path.join(path_distill, "train_probabilities.txt"), "rb")
    distill_train_pro_dict = pickle.load(distill_train_pro_file, encoding = "latin1")
    train_id_pro = { ID: [branch_train_id_pro[ID][0], branch_train_id_pro[ID][1] + probabilities] for ID, probabilities in zip(distill_train_pro_dict["ID"], distill_train_pro_dict["Probabilities"]) if ID in train_id_label}

    distill_dev_pro_file = open(os.path.join(path_distill, "dev_probabilities.txt"), "rb")
    distill_dev_pro_dict = pickle.load(distill_dev_pro_file, encoding = "latin1")
    dev_id_pro = { ID:[branch_dev_id_pro[ID][0], branch_dev_id_pro[ID][1] + probabilities] for ID, probabilities in zip(distill_dev_pro_dict["ID"], distill_dev_pro_dict["Probabilities"]) if ID in dev_id_label}

    distill_test_pro_file = open(os.path.join(path_distill, "test_probabilities.txt"), "rb")
    distill_test_pro_dict = pickle.load(distill_test_pro_file, encoding = "latin1")
    test_id_pro = { ID:[branch_test_id_pro[ID][0], branch_test_id_pro[ID][1] + probabilities] for ID, probabilities in zip(distill_test_pro_dict["ID"], distill_test_pro_dict["Probabilities"]) if ID in test_id_label}

    input_data = [train_id_pro, dev_id_pro, test_id_pro]
    return input_data

def voting_classifier(input_data, fold_num):
    train_X = []
    train_Y = []
    train_ids = []
    test_X = []
    test_Y = []
    test_ids = []

    for i in range(len(input_data)):
        if i < 2:
            for tweetid, label_pro_list in input_data[i].items():
                train_X.append(label_pro_list[1])
                train_Y.append(label_pro_list[0])
                train_ids.append(tweetid)
        else:
            for tweetid, label_pro_list in input_data[i].items():
                test_X.append(label_pro_list[1])
                test_Y.append(label_pro_list[0])
                test_ids.append(tweetid)

    param_grid = {'kernel':('rbf', 'linear'),'C':[0.8, 0.85, 0.90, 0.95, 1], 'gamma':[0.045, 0.05, 0.055, 0.06, 0.065]}
    grid_search = GridSearchCV(SVC(decision_function_shape = 'ovo'), param_grid)
    grid_search.fit(train_X, train_Y)
    best_params = grid_search.best_params_
    #print ("fold %s best parameters:" % str(fold_num), grid_search.best_params_)
    predicted_labels = grid_search.predict(np.array(test_X))

    out_path = "./voting_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num2label = {'0':'support', '1':'query', '2':'deny', '3':'comment'}
    values = [num2label[str(i)] for i in predicted_labels] 
    result_dictionary = dict(zip(test_ids, values))
    with open(os.path.join(out_path,'prediction_fold%s.txt' % str(fold_num)), 'w+') as outfile:
        json.dump(result_dictionary, outfile)
    print ("saved result and predictions for fold%s" % str(fold_num))
    outfile.close()

    with open(os.path.join(out_path, "best_parameters.txt"), 'a') as outfile:
        outfile.write("fold %s: " % str(fold_num))
        json.dump(grid_search.best_params_, outfile)
        outfile.write("\n")
    print ("saved best parameters for fold%s" % str(fold_num))
    outfile.close()

    if best_params['kernel'] == 'linear':
        #feature_weights = grid_search.coef_()
        clf = SVC(kernel = best_params['kernel'], C = best_params['C'], gamma = best_params['gamma'])
        feature_weights = clf.coef_
        print ("fold %s feature weights(linear kernel):" % str(fold_num), feature_weights)
        with open(os.path.join(out_path, "feature_weights.txt"), 'a') as outfile:
            outfile.write("fold %s: " % str(fold_num))
            json.dump(feature_weights, outfile)
            outfile.write("\n")
        print ("saved feature weigths for fold%s(linear kernel)" % str(fold_num))
        outfile.close()

if __name__ == "__main__":
    fold_num = sys.argv[1]
    BranchLSTM_pro_path = "../branchLSTM_cross_validation/saved_data_new/fold%s" % str(fold_num)
    DistilBert_pro_path = "../BERT_baseline/onestep_classification/saved_probability/fold%s" % str(fold_num)

    train_dev_split = load_dataset()
    train_dev_splits = Cross_validation_threads(train_dev_split)
    dataset = Extract_dataset(train_dev_splits[int(fold_num)])

    Input_data = preprocess_probabilities(BranchLSTM_pro_path, DistilBert_pro_path, dataset)

    voting_classifier(Input_data, fold_num)
