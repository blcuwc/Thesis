# -*- coding: utf-8 -*-
#environment : python3.6

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
import sklearn_crfsuite
import scipy.stats
import warnings
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from preprocessing import load_dataset, Cross_validation_threads
from Classify import Extract_dataset

def save_feature_weights(best_params, train_X, train_Y, fold_num):
    out_path = "./crf_svm_voting_output/"

    categories = ['support', 'deny', 'query']
    clf = SVC(kernel = 'linear', decision_function_shape='ovr', C = best_params['C'], gamma = best_params['gamma'])
    clf.fit(train_X, train_Y)
    coef_array = np.array(clf.coef_)
    print ("coefficients shape:", np.shape(coef_array))

    if os.path.exists(os.path.join(out_path, 'feature_weights.npy')):
        feature_weights_array = np.load(os.path.join(out_path, 'feature_weights.npy'))
        coef_array = feature_weights_array + coef_array
        np.save(os.path.join(out_path, 'feature_weights.npy'), coef_array)
    else:
        np.save(os.path.join(out_path, 'feature_weights.npy'), coef_array)


    weight_index = []
    weight_columns = ['support', 'comment', 'deny', 'query', 'support', 'comment', 'deny', 'query']
    for i in range(len(categories) - 1):
        for j in range(i+1, len(categories)):
            weight_index.append((categories[i], categories[j]))
    feature_weights_df = pd.DataFrame(coef_array, index = weight_index, columns = weight_columns)
    print ("fold %s feature weights sum:\n" % str(fold_num), feature_weights_df.to_string())

    with open(os.path.join(out_path, "feature_weights.txt"), 'w') as outfile:
        print ("fold %s feature weights:\n" % str(fold_num), feature_weights_df.to_string(), file=outfile)
    print ("saved feature weigths")
    outfile.close()

    with open(os.path.join(out_path, "feature_weights_latex.txt"), 'w') as outfile:
        print ("fold %s feature weights latex text:\n" % str(fold_num), feature_weights_df.to_latex(), file=outfile)
    print ("saved feature weigths to latex")
    outfile.close()

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
        return(['0', '0'])
    elif label == "comment":
        return(['1', '1'])
    elif label == "deny":
        return(['2', '0'])
    elif label == "query":
        return(['3', '0'])
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
    label_list_real = {}
    label_list_real['train'] = []
    label_list_real['dev'] = []
    label_list_real['test'] = []
    label_list_binary = {}
    label_list_binary['train'] = []
    label_list_binary['dev'] = []
    label_list_binary['test'] = []

    # also store IDs
    ID_list = {}
    ID_list['train'] = []
    ID_list['dev'] = []
    ID_list['test'] = []

    for sset in whichset:
        for conversation in train_dev_split[sset]:

            for branch in conversation['branches']:
                branch_rep = []  # list of all tweets in the branch
                temp_label_real = []
                temp_label_binary = []
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
                    temp_label_real.append(convertlabel(label)[0])  # convertlabel
                    temp_label_binary.append(convertlabel(label)[1])  # convertlabel
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
                label_list_real[sset].append(temp_label_real)
                label_list_binary[sset].append(temp_label_binary)
    return branch_list, ID_list, [label_list_real, label_list_binary]

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

def First_step_voting(branch_list, ID_list, label_list, fold_num):
    X_train = branch_list['train'] + branch_list['dev']
    y_train_real = label_list[0]['train'] + label_list[0]['dev']
    y_train_binary = label_list[1]['train'] + label_list[1]['dev']

    X_test = branch_list['test']
    y_test_real = label_list[0]['test']
    y_test_binary = label_list[1]['test']

    print ("X_train[0][0]:", (X_train[0][0]))
    print ("y_train[0]:", (y_train_binary[0]))
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    params_space = {'c1':scipy.stats.expon(scale=0.5), 'c2':scipy.stats.expon(scale=0.05)}
    rs = RandomizedSearchCV(crf, params_space, n_iter=20, n_jobs=1, cv=3, verbose=1)

    rs.fit(X_train, y_train_binary)
    y_pred = rs.predict(X_test)
    print('crf best params:', rs.best_params_)
    print('crf best CV score:', rs.best_score_)
    #print (metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=['0','1', '2', '3']))

    #save first step predicted labels
    #save training data for second step classification
    num2label = {'0':'non-comment', '1':'comment'}
    num2reallabel = {'0':'support', '1':'comment', '2':'deny', '3':'query'}
    first_result_dictionary = {}
    second_ID_list = []
    second_label_list = []
    #second_feature_list = []
    for i in range(len(y_test_binary)):
        for j in range(len(y_test_binary[i])):
            test_binary_label = y_test_binary[i][j]
            test_real_label = y_test_real[i][j]
            pred_label = y_pred[i][j]
            if ID_list['test'][i][j] in first_result_dictionary:
                continue
            else:
                if num2label[pred_label] == 'comment':
                    first_result_dictionary[ID_list['test'][i][j]] = num2label[pred_label]
                elif num2label[pred_label] == 'non-comment':
                    second_ID_list.append(ID_list['test'][i][j])
                    second_label_list.append(num2reallabel[test_real_label])
                    #second_feature_list.append()

    return second_ID_list, second_label_list, first_result_dictionary

def Second_step_voting(test_ID_list, test_label_list, probability_data, dataset_dict):
    X_train = []
    y_train = []
    X_test = []
    y_test = test_label_list

    for ID in test_ID_list:
        X_test.append(probability_data['test'][ID])

    for dataset in ['train', 'dev']:
        for i in range(len(dataset_dict[dataset][1])):
            label = dataset_dict[dataset][1][i]
            if label != 'comment':
                y_train.append(label)
                tweet_id = dataset_dict[dataset][2][i]
                X_train.append(probability_data[dataset][tweet_id])

    print ("shape train_X:", np.shape(np.array(X_train)))
    print ("shape train_Y:", np.shape(np.array(y_train)))
    print ("classes:", np.unique(np.array(y_train)))
    print ("max probability in train_X:", np.max(np.array(X_train)))
    print ("min probability in train_X:", np.min(np.array(X_train)))
    param_grid = {'kernel':('rbf', 'linear'),'C':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'gamma':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}
    grid_search = GridSearchCV(SVC(kernel='linear', decision_function_shape='ovr'), param_grid)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    #print ("fold %s best parameters:" % str(fold_num), grid_search.best_params_)
    predicted_labels = grid_search.predict(np.array(X_test))

    #save predicted labels
    #num2label = {'0':'support', '2':'deny', '3':'query'}
    #values = [num2label[str(i)] for i in predicted_labels] 
    result_dictionary = dict(zip(test_ID_list, predicted_labels))
    save_feature_weights(best_params, X_train, y_train, fold_num)

    return result_dictionary

def save_classification_result(first_result_dictionary, second_result_dictionary, fold_num):
    out_path = "./crf_svm_voting_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    first_result_dictionary.update(second_result_dictionary)
    with open(os.path.join(out_path,'prediction_fold%s.txt' % str(fold_num)), 'w+') as outfile:
        json.dump(first_result_dictionary, outfile)
    print ("saved result and predictions for fold%s" % str(fold_num))
    outfile.close()

if __name__ == "__main__":
    fold_num = sys.argv[1]
    BranchLSTM_pro_path = "../Adapted_branchLSTM/saved_data_new/fold%s" % str(fold_num)
    DistilBert_pro_path = "../BERT/onestep_classification/saved_probability/fold%s" % str(fold_num)

    train_dev_split = load_dataset()
    train_dev_splits = Cross_validation_threads(train_dev_split)
    dataset = Extract_dataset(train_dev_splits[int(fold_num)])

    # probability_data['dataset_name'] = {'tweet_id':[connected_probability]}
    probability_data = Load_probabilities(BranchLSTM_pro_path, DistilBert_pro_path, dataset)

    #branch_list = [[branch_feature_1], ..., [branch_feature_n]]
    #ID_list = [[branch_ID_1], ..., [branch_ID_n]]
    #label_list = [{'train':[[branch_reallabel1], ..., [branch_reallabel_n]], 'dev':, 'test':}, {'train'[[branch_binarylabel_1], ...,[]]:, 'dev':, 'test':}]
    branch_list, ID_list, label_list = preprocess_data(train_dev_splits, fold_num, probability_data)

    #second_label_list: second step classification test dataset labels, with only three classes
    second_ID_list, second_label_list, first_result_dictionary = First_step_voting(branch_list, ID_list, label_list, fold_num)
    second_result_dictionary = Second_step_voting(second_ID_list, second_label_list, probability_data, dataset)
    save_classification_result(first_result_dictionary, second_result_dictionary, fold_num)
