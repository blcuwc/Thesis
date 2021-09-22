# -*- coding: utf-8 -*-
#environment : ktrain

import sys
import sklearn
import pickle
import os
import numpy as np
import pandas as pd
import json
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from preprocessing import load_dataset, Cross_validation_threads
from Classify import Extract_dataset
import warnings

def Load_probabilities(path_branch, path_distill, dataset_dict):
    #dataset_dict[dataset_name] = [tweet_list, label_list, id_list]
    datasets = ['train', 'dev', 'test']
    #labels = ['support', 'comment', 'deny', 'query']
    labels2num = {'support':0, 'comment':1, 'deny':2, 'query':3}
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
                id_pro[ID] = [labels2num[label], connected_probability]

        input_data[dataset] = id_pro
    return input_data

def save_feature_weights(best_params, train_X, train_Y, fold_num):
    out_path = "./svm_voting_output/"

    categories = ['support', 'comment', 'deny', 'query']
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


def voting_classifier(input_data, fold_num):
    datasets = ['train', 'dev', 'test']
    train_X = []
    train_Y = []
    train_ids = []
    test_X = []
    test_Y = []
    test_ids = []

    for dataset in datasets:
        if dataset == 'train' or dataset == 'dev':
            for tweetid, label_pro_list in input_data[dataset].items():
                train_X.append(label_pro_list[1])
                train_Y.append(label_pro_list[0])
                train_ids.append(tweetid)
        else:
            for tweetid, label_pro_list in input_data[dataset].items():
                test_X.append(label_pro_list[1])
                test_Y.append(label_pro_list[0])
                test_ids.append(tweetid)

    print ("shape train_X:", np.shape(np.array(train_X)))
    print ("shape train_Y:", np.shape(np.array(train_Y)))
    print ("max probability in train_X:", np.max(np.array(train_X)))
    print ("min probability in train_X:", np.min(np.array(train_X)))
    param_grid = {'C':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'gamma':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}
    grid_search = GridSearchCV(SVC(kernel='linear', decision_function_shape='ovr'), param_grid)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid_search.fit(train_X, train_Y)
    best_params = grid_search.best_params_
    #print ("fold %s best parameters:" % str(fold_num), grid_search.best_params_)
    predicted_labels = grid_search.predict(np.array(test_X))

    out_path = "./svm_voting_output/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #save predicted labels
    num2label = {'0':'support', '1':'comment', '2':'deny', '3':'query'}
    values = [num2label[str(i)] for i in predicted_labels] 
    result_dictionary = dict(zip(test_ids, values))
    with open(os.path.join(out_path,'prediction_fold%s.txt' % str(fold_num)), 'w+') as outfile:
        json.dump(result_dictionary, outfile)
    print ("saved result and predictions for fold%s" % str(fold_num))
    outfile.close()

    #save best parameters
    with open(os.path.join(out_path, "best_parameters.txt"), 'a') as outfile:
        outfile.write("fold %s: " % str(fold_num))
        json.dump(best_params, outfile)
        outfile.write("\n")
    print ("saved best parameters for fold%s" % str(fold_num))
    outfile.close()

    # save feature weights if kernel = linear
    save_feature_weights(best_params, train_X, train_Y, fold_num)

if __name__ == "__main__":
    fold_num = sys.argv[1]
    BranchLSTM_pro_path = "../Adapted_branchLSTM/saved_data_new/fold%s" % str(fold_num)
    DistilBert_pro_path = "../BERT/onestep_classification/saved_probability/fold%s" % str(fold_num)

    train_dev_split = load_dataset()
    train_dev_splits = Cross_validation_threads(train_dev_split)
    dataset = Extract_dataset(train_dev_splits[int(fold_num)])

    Input_data = Load_probabilities(BranchLSTM_pro_path, DistilBert_pro_path, dataset)

    print ("Fold %s Voting classification!" % str(fold_num))
    voting_classifier(Input_data, fold_num)
