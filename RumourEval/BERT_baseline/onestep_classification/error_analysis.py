# -*- coding: utf-8 -*-

import sys
import os
import json
import pickle
from preprocessing import load_dataset, Cross_validation_threads

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
    submission_file = os.path.join("output", "prediction_fold%s.txt" % str(fold_num))
    submission = json.load(open(submission_file, 'r'))
    return submission

if __name__ == "__main__":

    # Read fold num for error analysis
    fold_num = sys.argv[1]
    print ("Error analysis fold: %s" % str(fold_num))

    train_dev_split = load_dataset()

    train_dev_splits = Cross_validation_threads(train_dev_split)

    #dataset[dataset_name] = {tweet_id : [tweet_text, tweet_label]}
    dataset = Extract_dataset(train_dev_splits[int(fold_num)])

    #predictions = {tweet_id : predicted_label}
    predictions = get_predictions(fold_num)

    #dataset['test'] = {tweet_id : [text, true_label, predicted_label]}
    for tweet_id, predicted_label in predictions.items():
        if tweet_id in list(dataset['test'].keys()):
            dataset['test'][tweet_id].append(predicted_label)

    print ("tweet_id\ttweet_text\ttrue_label\tpredicted_label")
    for tweet_id, info_list in dataset['test'].items():
        print (tweet_id + '\t' + info_list[0] + '\t' + info_list[1] + '\t' + info_list[2])

