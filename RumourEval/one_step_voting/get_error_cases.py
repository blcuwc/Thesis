# -*- coding: utf-8 -*-

import sys
import os
import json
import pickle
from preprocessing import load_dataset, Cross_validation_threads
reload(sys)
sys.setdefaultencoding('utf8')

def Extract_dataset(train_dev_split):
    dataset = {}
    for dataset_name, con_list in train_dev_split.items():
        tweet_info = {}
        for conversation in con_list:
            source_tweet = conversation['source']
            replies = conversation['replies']
            if 'label' not in source_tweet.keys() or source_tweet['text'] == None:
                pass
            else:
                tweet_info[source_tweet['id_str']] = [source_tweet['text'], source_tweet['label']]
            for reply_tweet in replies:
                if 'label' not in reply_tweet.keys() or reply_tweet['text'] == None:
                    continue
                else:
                    tweet_info[reply_tweet['id_str']] = [reply_tweet['text'], reply_tweet['label']]
        dataset[dataset_name] = tweet_info
        #print ("dataset name:", dataset_name)
        #print ("tweet list:", dataset[dataset_name][0])
        #print ("label list:", dataset[dataset_name][1])
        #print (dataset.keys())
    return dataset

def get_predictions(fold_num):
    crf_submission_file = os.path.join("crf_voting_output", "prediction_fold%s.txt" % str(fold_num))
    crf_submission = json.load(open(crf_submission_file, 'r'))
    svm_submission_file = os.path.join("svm_voting_output", "prediction_fold%s.txt" % str(fold_num))
    svm_submission = json.load(open(svm_submission_file, 'r'))
    return crf_submission, svm_submission

def print_fp_fn(dataset, fold_num, label_name, classifier):
    if not os.path.exists("%s_error_cases" % classifier):
        os.mkdir("%s_error_cases" % classifier)
    # print commnet false positive samples
    fp_error_file = open("%s_error_cases/%s_fp_error_fold%s" % (classifier, label_name, fold_num), 'w+')
    fp_error_file.write("number\ttweet_id\ttweet_text\ttrue_label\tpredicted_label\n")
    i = 0
    for tweet_id, info_list in dataset['test'].items():
        if info_list[1] != label_name and info_list[2] == label_name:
            info_list[0] = info_list[0].replace('\n', ' ')
            fp_error_file.write(str(i) + '\t' + tweet_id + '\t' + info_list[0] + '\t' + info_list[1] + '\t' + info_list[2] + '\n')
            i += 1

    fn_error_file = open("%s_error_cases/%s_fn_error_fold%s" % (classifier, label_name, fold_num), 'w+')
    fn_error_file.write("number\ttweet_id\ttweet_text\ttrue_label\tpredicted_label\n")
    i = 0
    # print comment false negative samples
    for tweet_id, info_list in dataset['test'].items():
        if info_list[1] == label_name and info_list[2] != label_name:
            info_list[0] = info_list[0].replace('\n', ' ')
            fn_error_file.write(str(i) + '\t' + tweet_id + '\t' + info_list[0] + '\t' + info_list[1] + '\t' + info_list[2] + '\n')
            i += 1

if __name__ == "__main__":

    # Read fold num for error analysis
    fold_num = sys.argv[1]
    print ("Error analysis fold: %s" % str(fold_num))

    train_dev_split = load_dataset()

    train_dev_splits = Cross_validation_threads(train_dev_split)

    #dataset[dataset_name] = {tweet_id : [tweet_text, tweet_label]}
    dataset = Extract_dataset(train_dev_splits[int(fold_num)])
    dataset_copy = Extract_dataset(train_dev_splits[int(fold_num)])
    #dataset_copy = dataset.copy()

    #predictions = {tweet_id : predicted_label}
    crf_predictions, svm_predictions = get_predictions(fold_num)

    #dataset['test'] = {tweet_id : [text, true_label, predicted_label]}
    # get crf predictions
    for tweet_id, predicted_label in crf_predictions.items():
        if tweet_id in list(dataset['test'].keys()):
            dataset['test'][tweet_id].append(predicted_label)

    # get svm predictions 
    for tweet_id, predicted_label in svm_predictions.items():
        if tweet_id in list(dataset_copy['test'].keys()):
            dataset_copy['test'][tweet_id].append(predicted_label)
    
    print_fp_fn(dataset, fold_num, 'comment', 'crf')
    print_fp_fn(dataset, fold_num, 'support', 'crf')
    print_fp_fn(dataset, fold_num, 'deny', 'crf')
    print_fp_fn(dataset, fold_num, 'query', 'crf')

    print_fp_fn(dataset_copy, fold_num, 'comment', 'svm')
    print_fp_fn(dataset_copy, fold_num, 'support', 'svm')
    print_fp_fn(dataset_copy, fold_num, 'deny', 'svm')
    print_fp_fn(dataset_copy, fold_num, 'query', 'svm')
