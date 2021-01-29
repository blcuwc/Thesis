# -*- coding: utf-8 -*-
#environment : python3.6

import sys
import os
import pickle
import json
import numpy as np
import sklearn_crfsuite
import scipy.stats
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from preprocessing import load_dataset, Cross_validation_threads
from Classify import Extract_dataset

def twpro2features(tw, i, branch, conversation, probability_data):
    #probability_data = {tweet_id : [num_label, [connected_probability]]}
    tweetid = branch[i]

    features = {
        'branchLSTM_support':probability_data[tweetid][0],
        'branchLSTM_comment':probability_data[tweetid][1],
        'branchLSTM_deny':probability_data[tweetid][2],
        'branchLSTM_query':probability_data[tweetid][3],
        'DistilBert_support':probability_data[tweetid][4],
        'DistilBert_comment':probability_data[tweetid][5],
        'DistilBert_deny':probability_data[tweetid][6],
        'DistilBert_query':probability_data[tweetid][7],
    }

    if i > 0:
        prevtweet_id = branch[i-1]
        if prevtweet_id in probability_data:
            features.update({
                '-1:branchLSTM_support':probability_data[prevtweet_id][0],
                '-1:branchLSTM_comment':probability_data[prevtweet_id][1],
                '-1:branchLSTM_deny':probability_data[prevtweet_id][2],
                '-1:branchLSTM_query':probability_data[prevtweet_id][3],
                '-1:DistilBert_support':probability_data[prevtweet_id][4],
                '-1:DistilBert_comment':probability_data[prevtweet_id][5],
                '-1:DistilBert_deny':probability_data[prevtweet_id][6],
                '-1:DistilBert_query':probability_data[prevtweet_id][7],
                
            })
    else:
        features['BOS'] = True

    if i < len(branch) - 1:
        latertweet_id = branch[i+1]
        if latertweet_id in probability_data:
            features.update({
                '+1:branchLSTM_support':probability_data[latertweet_id][0],
                '+1:branchLSTM_comment':probability_data[latertweet_id][1],
                '+1:branchLSTM_deny':probability_data[latertweet_id][2],
                '+1:branchLSTM_query':probability_data[latertweet_id][3],
                '+1:DistilBert_support':probability_data[latertweet_id][4],
                '+1:DistilBert_comment':probability_data[latertweet_id][5],
                '+1:DistilBert_deny':probability_data[latertweet_id][6],
                '+1:DistilBert_query':probability_data[latertweet_id][7],
                
            })
    else:
        features['EOS'] = True

    return features

def convertlabel(label):
    # return [real_label, comment/non-comment label]
    if label == "support":
        return('0')
    elif label == "comment":
        return('1')
    elif label == "deny":
        return('2')
    elif label == "query":
        return('3')
    else:
        print(label)

def preprocess_data(train_dev_splits, fold_num, probability_data):
    #probability_data[dataset] = [connected_probability]

    train_dev_split = train_dev_splits[int(fold_num)]

    whichset = ['train', 'dev', 'test']

    # first put everything in dict contatining lists for each set
    branch_list = {}
    branch_list['train'] = []
    branch_list['dev'] = []
    branch_list['test'] = []

    # also store labels
    label_list = {}
    label_list['train'] = []
    label_list['dev'] = []
    label_list['test'] = []

    # also store IDs
    ID_list = {}
    ID_list['train'] = []
    ID_list['dev'] = []
    ID_list['test'] = []

    for sset in whichset:
        for conversation in train_dev_split[sset]:

            for branch in conversation['branches']:
                branch_rep = []  # list of all tweets in the branch
                temp_label = []
                temp_id = []
                for i, tweetid in enumerate(branch):
                    # find tweet instance
                    if i == 0:
                        tweet = conversation['source']
                    else:
                        # tweet = {}
                        for response in conversation['replies']:
                            if tweetid == response['id_str']:
                                tweet = response
                                break
                    label = tweet['label']
                    temp_label.append(convertlabel(label))  # convertlabel
                    temp_id.append(tweet['id_str'])

                    if tweet['used']:
                        # if tweet has been processed then take the representation
                        representation = tweet['representation']
                    else:
                        # if tweet is new then get tweet's representation
                        representation = twpro2features(tweet, i, branch, conversation, probability_data[sset])
                        tweet['representation'] = representation
                        tweet['used'] = 1
                    branch_rep.append(representation)

                branch_list[sset].append(branch_rep)
                ID_list[sset].append(temp_id)
                label_list[sset].append(temp_label)
    return branch_list, ID_list, label_list

def Load_probabilities(path_branch, path_distill, dataset_dict):
    #dataset_dict[dataset_name] = [tweet_list, label_list, id_list]
    datasets = ['train', 'dev', 'test']
    input_data = {}

    for dataset in datasets:
        pro_file1 = open(os.path.join(path_branch, dataset + "/probabilities.txt"), "rb")
        pro_file2 = open(os.path.join(path_distill, dataset + "_probabilities.txt"), "rb")
        pro_dict1 = pickle.load(pro_file1, encoding = "latin1")
        pro_dict2 = pickle.load(pro_file2, encoding = "latin1")

        id_pro1 = {}
        id_pro2 = {}
        id_pro = {}

        for ID, probabilities in zip(pro_dict1["ID"], pro_dict1["Probabilities"]):
            id_pro1[ID] = probabilities
        for ID, probabilities in zip(pro_dict2["ID"], pro_dict2["Probabilities"]):
            id_pro2[ID] = probabilities

        for ID, label in zip(dataset_dict[dataset][2], dataset_dict[dataset][1]):
            if ID in id_pro1 and ID in id_pro2:
                connected_probability = list(id_pro1[ID]) + list(id_pro2[ID])
                id_pro[ID] = connected_probability

        input_data[dataset] = id_pro
    return input_data

def crf_voting(branch_list, ID_list, label_list, fold_num):
    X_train = branch_list['train'] + branch_list['dev']
    y_train = label_list['train'] + label_list['dev']
    X_test = branch_list['test']
    y_test = label_list['test']

    print ("X_train[0][0]:", (X_train[0][0]))
    print ("y_train[0]:", (y_train[0]))
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    params_space = {'c1':scipy.stats.expon(scale=0.5), 'c2':scipy.stats.expon(scale=0.05)}
    rs = RandomizedSearchCV(crf, params_space, n_iter=20, n_jobs=1, cv=3, verbose=1)

    rs.fit(X_train, y_train)
    y_pred = rs.predict(X_test)
    print('crf best params:', rs.best_params_)
    print('crf best CV score:', rs.best_score_)
    #print (metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=['0','1', '2', '3']))

    #save predicted labels
    num2label = {'0':'support', '1':'comment', '2':'deny', '3':'query'}
    result_dictionary = {}
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            test_label = y_test[i][j]
            pred_label = y_pred[i][j]
            if ID_list['test'][i][j] in result_dictionary:
                continue
            else:
                result_dictionary[ID_list['test'][i][j]] = num2label[pred_label]

    out_path = "./crf_voting_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path,'prediction_fold%s.txt' % str(fold_num)), 'w+') as outfile:
        json.dump(result_dictionary, outfile)
    print ("saved result and predictions for fold%s" % str(fold_num))
    outfile.close()

if __name__ == "__main__":
    fold_num = sys.argv[1]
    BranchLSTM_pro_path = "../branchLSTM_cross_validation/saved_data_new/fold%s" % str(fold_num)
    DistilBert_pro_path = "../BERT_baseline/onestep_classification/saved_probability/fold%s" % str(fold_num)

    train_dev_split = load_dataset()
    train_dev_splits = Cross_validation_threads(train_dev_split)
    dataset = Extract_dataset(train_dev_splits[int(fold_num)])

    # probability_data['dataset_name'] = {'tweet_id':[connected_probability]}
    probability_data = Load_probabilities(BranchLSTM_pro_path, DistilBert_pro_path, dataset)

    #branch_list = [[branch_feature_1], ..., [branch_feature_n]]
    #ID_list = [[branch_ID_1], ..., [branch_ID_n]]
    #label_list = {'train':[[branch_label1], ..., [branch_label_n]], 'dev':, 'test':}
    branch_list, ID_list, label_list = preprocess_data(train_dev_splits, fold_num, probability_data)

    crf_voting(branch_list, ID_list, label_list, fold_num)
