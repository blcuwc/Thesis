# -*- coding: utf-8 -*-
#environment : ktrain
# usage : CUDA_VISIBLE_DEVICES=x python stepone_classify.py > stepone_log
"""
This file contains preprocessing routines to convert RumourEval data 
into the format of branchLSTM input: it splits conversation trees into 
branches and extracts features from tweets including average of word2vec and 
extra features (specified in 
https://www.aclweb.org/anthology/S/S17/S17-2083.pdf) and concatenates them. 


Assumes that data is in the same folder as the script.
Dataset: http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools

Run:
    
python2 preprocessing.py

Saves processed data in saved_data folder

"""
import os
import numpy as np
import json
import gensim
import nltk
import re
from nltk.corpus import stopwords
from copy import deepcopy
import pickle
import ktrain
from ktrain import text
from sklearn.metrics import classification_report


def load_true_labels(dataset_name):

    # Training and development datasets should be stored in the downloaded_data folder (see installation instructions).
    # The test data is kept in the repo for now.
    traindev_path = os.path.join("../../branchLSTM/downloaded_data", "semeval2017-task8-dataset", "traindev")
    data_files = {"dev": os.path.join(traindev_path, "rumoureval-subtaskA-dev.json"),
                  "train": os.path.join(traindev_path, "rumoureval-subtaskA-train.json"),
                  "test": "../subtaska.json"}

    # Load the dictionary containing the tweets and labels from the .json file
    with open(data_files[dataset_name]) as f:
        for line in f:
            tweet_label_dict = json.loads(line)

    return tweet_label_dict


def load_dataset():

    # Load labels and split for task A
    dev = load_true_labels("dev")
    train = load_true_labels("train")
    test = load_true_labels("test")
    #get all tweets ids
    dev_tweets = dev.keys()
    train_tweets = train.keys()
    test_tweets = test.keys()

    # Load folds and conversations
    path_to_folds = os.path.join('../../branchLSTM/downloaded_data', 'semeval2017-task8-dataset/rumoureval-data')
    folds = sorted(os.listdir(path_to_folds))
    newfolds = [i for i in folds if i[0] != '.']
    folds = newfolds
    cvfolds = {}
    allconv = []
    weird_conv = []
    weird_struct = []
    train_dev_split = {}
    train_dev_split['dev'] = []
    train_dev_split['train'] = []
    train_dev_split['test'] = []
    for nfold, fold in enumerate(folds):
        path_to_tweets = os.path.join(path_to_folds, fold)
        tweet_data = sorted(os.listdir(path_to_tweets))
        newfolds = [i for i in tweet_data if i[0] != '.']
        tweet_data = newfolds
        conversation = {}
        for foldr in tweet_data:
            flag = 0
            conversation['id'] = foldr
            path_src = path_to_tweets+'/'+foldr+'/source-tweet'
            files_t = sorted(os.listdir(path_src))
            #open source tweet file
            with open(os.path.join(path_src, files_t[0])) as f:
                #just one line json file
                for line in f:
                    src = json.loads(line)
                    src['used'] = 0
                    scrcid = src['id_str']
                    # add set and label to tweet info
                    # first find the tweet in one of the sets
                    # foldr - src tweet id
                    if scrcid in dev_tweets:
                        src['set'] = 'dev'
                        src['label'] = dev[scrcid]
                        #flag shows source tweet is in dev dataset
                        flag = 'dev'
    #                    train_dev_tweets['dev'].append(src)
                    elif scrcid in train_tweets:
                        src['set'] = 'train'
                        src['label'] = train[scrcid]
                        #flag shows source tweet is in train dataset
                        flag = 'train'
    #                    train_dev_tweets['train'].append(src)
                    else:
                        src['set'] = 'Null'
                        print ("Tweet was not found! ID: ", foldr)
            conversation['source'] = src
            if src['text'] is None:
                print ("Tweet has no text", src['id'])
            tweets = []
            path_repl = path_to_tweets+'/'+foldr+'/replies'
            files_t = sorted(os.listdir(path_repl))
            newfolds = [i for i in files_t if i[0] != '.']
            files_t = newfolds
            for repl_file in files_t:
                with open(os.path.join(path_repl, repl_file)) as f:
                    for line in f:
                        tw = json.loads(line)
                        tw['used'] = 0
                        replyid = tw['id_str']
                        if replyid in dev_tweets:
                            tw['set'] = 'dev'
                            tw['label'] = dev[replyid]
    #                        train_dev_tweets['dev'].append(tw)
                            #source tweet in train dataset but reply tweet in dev dataset
                            if flag == 'train':
                                print ("The tree is split between sets", foldr)
                        elif replyid in train_tweets:
                            tw['set'] = 'train'
                            tw['label'] = train[replyid]
    #                        train_dev_tweets['train'].append(tw)
                            #source tweet in dev dataset but reply in train dataset
                            if flag == 'dev':
                                print ("The tree is split between sets", foldr)
                        else:
                            tw['set'] = 'Null'
                            print ("Tweet was not found! ID: ", foldr)
                        tweets.append(tw)
                        if tw['text'] is None:
                            print ("Tweet has no text", tw['id'])
            conversation['replies'] = tweets
            path_struct = path_to_tweets+'/'+foldr+'/structure.json'
            with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
            if len(struct) > 1:
                # print "Structure has more than one root"
                new_struct = {}
                new_struct[foldr] = struct[foldr]
                struct = new_struct
                weird_conv.append(conversation.copy())
                weird_struct.append(struct)
                # Take item from structure if key is same as source tweet id
            conversation['structure'] = struct
            #branches = tree2branches(conversation['structure'])
            #conversation['branches'] = branches
            train_dev_split[flag].append(conversation.copy())
            allconv.append(conversation.copy())
        cvfolds[fold] = allconv
        allconv = []

    # Load testing data
    path_to_test = os.path.join('../../branchLSTM/downloaded_data', 'semeval2017-task8-test-data')
    test_folders = sorted(os.listdir(path_to_test))
    newfolds = [i for i in test_folders if i[0] != '.']
    test_folders = newfolds
    conversation = {}
    for tfldr in test_folders:
        conversation['id'] = tfldr
        path_src = path_to_test+'/'+tfldr+'/source-tweet'
        files_t = sorted(os.listdir(path_src))
        with open(os.path.join(path_src, files_t[0])) as f:
            for line in f:
                src = json.loads(line)
                src['used'] = 0
                scrcid = src['id_str']
                # add set and label to tweet info
                # first find the tweet in one of the sets
                # foldr - src tweet id
                if scrcid in test_tweets:
                    src['set'] = 'test'
                    src['label'] = test[scrcid]
                    #flag shows source tweet is in dev dataset
                    flag = 'test'
                else:
                    src['set'] = 'Null'
                    print ("Tweet was not found! ID: ", foldr)
        conversation['source'] = src
        tweets = []
        path_repl = path_to_test+'/'+tfldr+'/replies'
        files_t = sorted(os.listdir(path_repl))
        newfolds = [i for i in files_t if i[0] != '.']
        files_t = newfolds
        for repl_file in files_t:
            with open(os.path.join(path_repl, repl_file)) as f:
                for line in f:
                    tw = json.loads(line)
                    tw['used'] = 0
                    replyid = tw['id_str']
                    if replyid in test_tweets:
                        tw['set'] = 'test'
                        tw['label'] = test[replyid]
                    else:
                        tw['set'] = 'Null'
                        print ("Tweet was not found! ID: ", foldr)
            tweets.append(tw)
        conversation['replies'] = tweets
        path_struct = path_to_test+'/'+tfldr+'/structure.json'
        with open(path_struct) as f:
            for line in f:
                struct = json.loads(line)
        conversation['structure'] = struct
        #branches = tree2branches(conversation['structure'])
        #conversation['branches'] = branches
        train_dev_split['test'].append(conversation.copy())

    return train_dev_split

def Extract_dataset(train_dev_split):
    dataset = {}
    for dataset_name, con_list in train_dev_split.items():
        tweet_list = []
        label_list = []
        id_list = []
        for conversation in con_list:
            source_tweet = conversation['source']
            replies = conversation['replies']
            if 'label' not in source_tweet.keys() or source_tweet['text'] == None:
                pass
            else:
                tweet_list.append(source_tweet['text'])
                label_list.append(source_tweet['label'])
                id_list.append(source_tweet['id_str'])
            for reply_tweet in replies:
                if 'label' not in reply_tweet.keys() or reply_tweet['text'] == None:
                    continue
                else:
                    tweet_list.append(reply_tweet['text'])
                    label_list.append(reply_tweet['label'])
                    id_list.append(reply_tweet['id_str'])
        dataset[dataset_name] = [tweet_list, label_list, id_list]
        #print ("dataset name:", dataset_name)
        #print ("tweet list:", dataset[dataset_name][0])
        #print ("label list:", dataset[dataset_name][1])
        #print (dataset.keys())
    return dataset

def save_predictions(id_list, predictions):
    out_path = './output'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    result_dictionary = dict(zip(id_list, predictions))

    with open(os.path.join(out_path,'predictions.txt'), 'w+') as outfile:
        json.dump(result_dictionary, outfile)
    print ("saved result and predictions")
    #stop = timeit.default_timer()
    #print ("Time: ",stop - start)

def First_step_classify(dataset):
    #dataset['dataset_name'] = [tweet_list, label_list, id_list]
    #categories = ['support', 'query', 'deny', 'comment']
    categories = ['other', 'comment']
    #print ("categories:", categories)
    #categories2num = {'support':0, 'query':1, 'deny':2, 'comment':3}
    categories2binary = {'support':'other', 'query':'other', 'deny':'other', 'comment':'comment'}
    for dataset_name, lists in dataset.items():
        binary_label_list = []
        for label in lists[1]:
            binary_label = categories2binary[label]
            binary_label_list.append(binary_label)
        #dataset['dataset_name'] = [tweet_list, label_list, id_list, [0 1] list]
        dataset[dataset_name].append(binary_label_list)

    x_train = dataset['train'][0]
    y_train = dataset['train'][3]
    x_dev = dataset['dev'][0]
    y_dev = dataset['dev'][3]
    test_texts = dataset['test'][0]
    test_labels = dataset['test'][3]

    trn, val, preproc = text.texts_from_array(x_train = x_train, y_train = y_train,
                                          x_test = x_dev, y_test = y_dev,
                                          class_names = categories,
                                          preprocess_mode = 'distilbert',
                                          maxlen = 350)
    print ("first step ktrain preprocess finished!")
    #text.print_text_classifiers()

    model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
    print ("first classifier built finished!")

    #learning rate and epochs
    learner.fit_onecycle(3e-5, 4)
    print ("first classifier training finished!")

    p = ktrain.get_predictor(model, preproc)

    predictions = []
    rest_test_dataset_texts = []
    rest_test_dataset_labels = []
    rest_test_dataset = {}

    for i in range(len(test_texts)):
        predictions.append(p.predict(test_texts[i]))
        if test_labels[i] == 'other' and predictions[i] == 'other':
            rest_test_dataset_texts.append(dataset['test'][0][i])
            rest_test_dataset_labels.append(dataset['test'][1][i])

    #print (classification_report(test_labels, predictions, list(np.unique(test_labels))))
    print ("*****************************")
    print ("First classification report: ")
    print ("*****************************")
    print (classification_report(test_labels, predictions, categories))

    rest_test_dataset['text'] = rest_test_dataset_texts
    rest_test_dataset['labels'] = rest_test_dataset_labels

    return dataset, rest_test_dataset, [predictions, test_labels]

def save_data(new_dataset, rest_test_dataset, first_predictions):
    output_path = "./output"

    with open(os.path.join(output_path, "dataset.txt"), "w") as outfile:
        json.dump(new_dataset, outfile)
    outfile.close()

    with open(os.path.join(output_path, "rest_test_dataset.txt"), "w") as outfile:
        json.dump(rest_test_dataset, outfile)
    outfile.close()

    with open(os.path.join(output_path, "first_predictions.txt"), "w") as outfile:
        json.dump(first_predictions, outfile)
    outfile.close()

if __name__ == "__main__":
    train_dev_split = load_dataset()
    dataset = Extract_dataset(train_dev_split)
    #print (dataset.keys())
    new_dataset, rest_test_dataset, first_predictions = First_step_classify(dataset)
    save_data(new_dataset, rest_test_dataset, first_predictions)
