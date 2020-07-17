# -*- coding: utf-8 -*-
#environemnt : ktrain
#usage: CUDA_VISIBLE_DEVICES=x python Classify.py > log
#predictions saved in ./output/predictions.txt
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
                  "test": "../../branchLSTM/subtaska.json"}

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
    allconv = []
    weird_conv = []
    weird_struct = []
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
                    if scrcid in dev_tweets:
                        src['label'] = dev[scrcid]
                    elif scrcid in train_tweets:
                        src['label'] = train[scrcid]
                    elif scrcid in test_tweets:
                        src['label'] = test[scrcid]
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
                            tw['label'] = dev[replyid]
                        elif replyid in train_tweets:
                            tw['label'] = train[replyid]
                        elif replyid in test_tweets:
                            tw['label'] = test[replyid]
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
                weird_conv.append(conversation)
                weird_struct.append(struct)
                # Take item from structure if key is same as source tweet id
            conversation['structure'] = struct
            branches = tree2branches(conversation['structure'])
            conversation['branches'] = branches
            allconv.append(conversation)

    # Load testing data
    path_to_test = os.path.join('../../branchLSTM/downloaded_data', 'semeval2017-task8-test-data')
    test_folders = sorted(os.listdir(path_to_test))
    newfolds = [i for i in test_folders if i[0] != '.']
    test_folders = newfolds
    conversation = {}
    for tfldr in test_folders:
        flag = 0
        conversation['id'] = tfldr
        path_src = path_to_test+'/'+tfldr+'/source-tweet'
        files_t = sorted(os.listdir(path_src))
        with open(os.path.join(path_src, files_t[0])) as f:
            for line in f:
                src = json.loads(line)
                src['used'] = 0
                srcid = src['id_str']
                if scrcid in dev_tweets:
                    src['label'] = dev[scrcid]
                elif scrcid in train_tweets:
                    src['label'] = train[scrcid]
                elif scrcid in test_tweets:
                    src['label'] = test[scrcid]
                else:
                    src['set'] = 'Null'
                    print ("Tweet was not found! ID: ", foldr)
        conversation['source'] = src
        if src['text'] is None:
            print ("Tweet has no text", src['id'])
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
                    if replyid in dev_tweets:
                        tw['label'] = dev[replyid]
                    elif replyid in train_tweets:
                        tw['label'] = train[replyid]
                    elif replyid in test_tweets:
                        tw['label'] = test[replyid]
                    else:
                        tw['set'] = 'Null'
                        print ("Tweet was not found! ID: ", foldr)
                    if tw['text'] is None:
                        print ("Tweet has no text", tw['id'])
            tweets.append(tw)
        conversation['replies'] = tweets
        path_struct = path_to_test+'/'+tfldr+'/structure.json'
        with open(path_struct) as f:
            for line in f:
                struct = json.loads(line)
        if len(struct) > 1:
            # print "Structure has more than one root"
            new_struct = {}
            new_struct[tfldr] = struct[tfldr]
            struct = new_struct
            weird_conv.append(conversation.copy())
            weird_struct.append(struct)
            # Take item from structure if key is same as source tweet id
        conversation['structure'] = struct
        branches = tree2branches(conversation['structure'])
        conversation['branches'] = branches
        allconv.append(conversation)

    return allconv

def tree2branches(root):
    node = root
    parent_tracker = []
    parent_tracker.append(root)
    branch = []
    branches = []
    i = 0
    while True:
        node_name = list(node.keys())[i]
        #print node_name
        branch.append(node_name)
        # get children of the node
        first_child = list(node.values())[i]
        # actually all chldren, all tree left under this node
        if first_child != []:  # if node has children
            node = first_child      # walk down
            parent_tracker.append(node)
            siblings = list(first_child.keys())
            i = 0  # index of a current node
        else:
            branches.append(deepcopy(branch))
            i = siblings.index(node_name)  # index of a current node
            # if the node doesn't have next siblings
            while i+1 >= len(siblings):
                if node is parent_tracker[0]:  # if it is a root node
                    return branches
                del parent_tracker[-1]
                del branch[-1]
                node = parent_tracker[-1]      # walk up ... one step
                node_name = branch[-1]
                siblings = list(node.keys())
                i = siblings.index(node_name)
            i = i+1    # ... walk right
#            node =  parent_tracker[-1].values()[i]
            del branch[-1]
#            branch.append(node.keys()[0])
#%%
# process tweet into features

def split_dataset(allconv):
    train_dev_splits = []
    train_dev_split = {}
    train_dev_split['dev'] = []
    train_dev_split['train'] = []
    train_dev_split['test'] = []
    np.random.seed(5)
    np.random.shuffle(allconv)

    folds = []
    fold_length = int(len(allconv)/5)
    for i in range(5):
        fold = allconv[i*fold_length:(i+1)*fold_length]
        folds.append(fold)
    folds[-1] += allconv[fold_length*5:len(allconv)]

    for i in range(len(folds)):
        #print ("folds"+ str(i), folds[i])
        train_dev_split['test'] = folds[i]
        train_dev_split['dev'] = folds[i-4]
        for fold in folds:
            if fold != folds[i] and fold != folds[i-4]:
                train_dev_split['train'] += fold
        #print ("train_dev_split:")
        #print (train_dev_split)
        train_dev_splits.append(train_dev_split)
        train_dev_split = {}
        train_dev_split['dev'] = []
        train_dev_split['train'] = []
        train_dev_split['test'] = []
        #print ("train_dev_splits:")
        #print (train_dev_splits)

    return train_dev_splits

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

def save_predictions(id_list, predictions, fold_num):
    out_path = './output_new'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    result_dictionary = dict(zip(id_list, predictions))

    with open(os.path.join(out_path,'prediction%s.txt' % str(fold_num)), 'w+') as outfile:
        json.dump(result_dictionary, outfile)
    print ("saved result and prediction%s!" % str(fold_num))

def save_test_labels(train_dev_splits):
    test_id_label = {}
    test_label_path = 'true_test_labels'
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)
    for fold_num in range(len(train_dev_splits)):
        print ("save %s test labels" % str(fold_num))
        test_conv = train_dev_splits[fold_num]['test']
        for conv in test_conv:
            src = conv['source']
            replies = conv["replies"]
            test_id_label[src['id_str']] = src['label']
            for reply in replies:
                test_id_label[reply['id_str']] = reply['label']
        with open(os.path.join(test_label_path,'fold%s.txt' % str(fold_num)), 'w+') as outfile:
            json.dump(test_id_label, outfile)
        test_id_label = {}

def Classify(dataset, fold_num):
    categories = ['support', 'query', 'deny', 'comment']
    #print ("categories:", categories)
    categories2num = {'support':0, 'query':1, 'deny':2, 'comment':3}

    x_train = dataset['train'][0]
    y_train = dataset['train'][1]
    x_dev = dataset['dev'][0]
    y_dev = dataset['dev'][1]
    test_texts = dataset['test'][0]
    test_labels = dataset['test'][1]

    trn, val, preproc = text.texts_from_array(x_train = x_train, y_train = y_train,
                                          x_test = x_dev, y_test = y_dev,
                                          class_names = categories,
                                          preprocess_mode = 'distilbert',
                                          maxlen = 350)
    print ("ktrain preprocess finished!")
    #text.print_text_classifiers()

    model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
    print ("distilbert built finished!")

    #learning rate and epochs
    learner.fit_onecycle(3e-5, 4)
    print ("distilbert training finished!")

    p = ktrain.get_predictor(model, preproc)

    p_test = []
    for tweet in test_texts:
        p_test.append(p.predict(tweet))

    #save predicted test dataset labels tweet_id:predicted_label
    save_predictions(dataset['test'][2], p_test, fold_num)

    print (classification_report(test_labels, p_test, list(np.unique(test_labels))))

if __name__ == "__main__":
    # get all conversations
    allconv = load_dataset()

    # 5-fold cross validation
    train_dev_splits = split_dataset(allconv)

    #save true test dataset labels tweet_id:true_label
    save_test_labels(train_dev_splits)

    for i in range(len(train_dev_splits)):
        dataset = Extract_dataset(train_dev_splits[i])
        #print (dataset.keys())
        print ("*********************************")
        print ("********** Fold %s **************" % str(i))
        print ("*********************************")
        Classify(dataset, i)
