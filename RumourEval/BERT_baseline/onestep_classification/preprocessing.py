# -*- coding: utf-8 -*-
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

import sys
import os
import numpy as np
import json
import gensim
import nltk
import re
from nltk.corpus import stopwords
from copy import deepcopy
import pickle


def load_true_labels(dataset_name):

    # Training and development datasets should be stored in the downloaded_data folder (see installation instructions).
    # The test data is kept in the repo for now.
    traindev_path = os.path.join("../../branchLSTM/downloaded_data", "semeval2017-task8-dataset", "traindev")
    data_files = {"dev": os.path.join(traindev_path, "rumoureval-subtaskA-dev.json"),
                  "train": os.path.join(traindev_path, "rumoureval-subtaskA-train.json"),
                  "test": "subtaska.json"}

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
                    elif scrid in test_tweets:
                        src['label'] = test[scrid]
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
                        elif replyid in test_tweets:
                            tw['set'] = 'test'
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
                weird_conv.append(conversation.copy())
                weird_struct.append(struct)
                # Take item from structure if key is same as source tweet id
            conversation['structure'] = struct
            branches = tree2branches(conversation['structure'])
            conversation['branches'] = branches
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
        conversation['structure'] = struct
        branches = tree2branches(conversation['structure'])
        conversation['branches'] = branches
        train_dev_split['test'].append(conversation.copy())

    return train_dev_split


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


def cleantweet(tweettext, tweet):
    #  for hashtag in tweet["entities"]["hashtags"]:
    #    tweettext = tweettext.replace(hashtag["text"], "")
    if "media" in tweet["entities"]:
        for url in tweet["entities"]["media"]:
            tweettext = tweettext.replace(url["url"], "picpicpic")
    if "urls" in tweet["entities"]:
        for url in tweet["entities"]["urls"]:
            tweettext = tweettext.replace(url["url"], "urlurlurl")
#  for usermention in tweet["entities"]["user_mentions"]:
#    tweettext = tweettext.replace(usermention["screen_name"], "")
    return tweettext
# converts sentece to list of tokens/words


def str_to_wordlist(tweettext, tweet, remove_stopwords=False):

    #  Remove non-letters
    # NOTE: Is it helpful or not to remove non-letters?
    # str_text = re.sub("[^a-zA-Z]"," ", str_text)
    tweettext = cleantweet(tweettext, tweet)
    str_text = re.sub("[^a-zA-Z]", " ", tweettext)
    # Convert words to lower case and split them
    # words = str_text.lower().split()
    words = nltk.word_tokenize(str_text.lower())
    # Optionally remove stop words (false by default)
    # NOTE: generic list of stop words, should i remove them or not?
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    # 5. Return a list of words
    return(words)
#%%
# Turn tweet into average of word vectors


def loadW2vModel():
    # LOAD PRETRAINED MODEL
    print ("Loading the model")
    model = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join('../../branchLSTM/downloaded_data', 'GoogleNews-vectors-negative300.bin'), binary=True)
    print ("Done!")
    return model


def sumw2v(model, tweet, avg=True):
    num_features = 300
    temp_rep = np.zeros(num_features)
    wordlist = str_to_wordlist(tweet['text'], tweet, remove_stopwords=False)
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep += model[wordlist[w]]
    if avg:
        sumw2v = temp_rep/len(wordlist)
    else:
        # sum
        sumw2v = temp_rep
    return sumw2v


def getW2vCosineSimilarity(model, words, wordssrc):
    words2 = []
    for word in words:
        if word in model.vocab:  # change to model.wv.vocab
            words2.append(word)
    wordssrc2 = []
    for word in wordssrc:
        if word in model.vocab:  # change to model.wv.vocab
            wordssrc2.append(word)

    if len(words2) > 0 and len(wordssrc2) > 0:
        return model.n_similarity(words2, wordssrc2)
    return 0.
#%%


def tweet2features(model, tw, i, branch, conversation):

    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+',
                                       '', tw['text'].lower()))
    srctweet = conversation['source']['text']
    if i > 0:
        prevtweet_id = branch[i-1]
        if (i-1) == 0:
            prevtweet = srctweet
        else:
            for response in conversation['replies']:
                if prevtweet_id == response['id_str']:
                    prevtweet = response['text']
                    break
        srctokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+',
                                              '', srctweet.lower()))
        prevtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+',
                                               '', prevtweet.lower()))
    otherthreadtweets = ''
    if i != 0:
        otherthreadtweets += srctweet
    for response in conversation['replies']:
        if response['user']['screen_name'] != tw['user']['screen_name']:
            otherthreadtweets += ' ' + response['text']

    otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+',
                                                  '',
                                                  otherthreadtweets.lower()))
    features = []
    tw['text'] = cleantweet(tw['text'], tw)
    issourcetw = int(tw['in_reply_to_screen_name'] == None)
    hasqmark = 0
    if tw['text'].find('?') >= 0:
        hasqmark = 1
    hasemark = 0
    if tw['text'].find('!') >= 0:
        hasemark = 1
    hasperiod = 0
    if tw['text'].find('.') >= 0:
        hasperiod = 0
    hasurl = 0
    if tw['text'].find('urlurlurl') >= 0 or tw['text'].find('http') >= 0:
        hasurl = 1
    haspic = 0
    if (tw['text'].find('picpicpic') >= 0) or (
            tw['text'].find('pic.twitter.com') >= 0) or ( 
                    tw['text'].find('instagr.am') >= 0):
        haspic = 1

    hasnegation = 0
    negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
                     'neither', 'nor', 'nowhere', 'hardly',
                     'scarcely', 'barely', 'don', 'isn', 'wasn',
                     'shouldn', 'wouldn', 'couldn', 'doesn']
    for negationword in negationwords:
        if negationword in tokens:
            hasnegation += 1

    # Character count using len(text) depends on whether a wide or narrow build of Python was used.
    # Let's set the count to be equivalent to that computed on a wide build, i.e. all unicode characters are counted as
    # length 1.
    tmp_tw_text = tw["text"].encode('raw_unicode_escape')
    tmp_tw_text = re.sub("(\\\\U[0-9A-Fa-f]{8})", "U", tmp_tw_text)
    tmp_tw_text = re.sub("(\\\\u[0-9A-Fa-f]{4})", "U", tmp_tw_text)
    charcount = len(tmp_tw_text)

    # To print the character counts before and after this change, along with the tweets themselves, uncomment below
    # print len(tw["text"]), "\t", charcount, \
    #       "\t\t", tw["text"],\
    #       "\t\t", tw["text"].encode('raw_unicode_escape'), \
    #       "\t\t", tmp_tw_text

    wordcount = len(nltk.word_tokenize(re.sub(r'([^\s\w]|_)+',
                                              '',
                                              tw['text'].lower())))

    swearwords = []
    with open('badwords.txt', 'r') as f:
        for line in f:
            swearwords.append(line.strip().lower())

    hasswearwords = 0
    for token in tokens:
        if token in swearwords:
            hasswearwords += 1
    uppers = [l for l in tw['text'] if l.isupper()]
    capitalratio = len(uppers)/len(tw['text'])

#%%
# W2vSimilarity wrt prev, thread, src
    if i > 0:
        Word2VecSimilarityWrtSource = getW2vCosineSimilarity(model, tokens, srctokens)
        Word2VecSimilarityWrtPrev = getW2vCosineSimilarity(model, tokens, prevtokens)
    else:
        Word2VecSimilarityWrtSource = 0
        Word2VecSimilarityWrtPrev = 0
    Word2VecSimilarityWrtOther = getW2vCosineSimilarity(model, tokens,
                                                        otherthreadtokens)

#%%
    avgw2v = sumw2v(model, tw, avg=True)
    features = [charcount, wordcount, issourcetw, hasqmark, hasemark,
                hasperiod, hasurl, haspic, hasnegation, hasswearwords,
                capitalratio, Word2VecSimilarityWrtSource,
                Word2VecSimilarityWrtPrev, Word2VecSimilarityWrtOther]
    features.extend(avgw2v)
    features = np.asarray(features, dtype=np.float32)
    return features
#%%

def convertlabel(label):
    if label == "support":
        return(0)
    elif label == "comment":
        return(1)
    elif label == "deny":
        return(2)
    elif label == "query":
        return(3)
    else:
        print(label)

def preprocess_data(train_dev_splits, fold_num):
    # Create train X, train Y, dev X, dev Y

    #%%
    model = loadW2vModel()
    #find max branch length
    train_dev_split = train_dev_splits[int(fold_num)]

    max_branch_len = {}
    max_branch_len['train'] = 0
    max_branch_len['dev'] = 0
    max_branch_len['test'] = 0

    whichset = ['train', 'dev', 'test']
    special = []

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

    rmdoublemask_list = {}
    rmdoublemask_list['train'] = []
    rmdoublemask_list['dev'] = []
    rmdoublemask_list['test'] = []

    dumplabel = {}
    dumplabel['train'] = []
    dumplabel['dev'] = []
    dumplabel['test'] = []

    for sset in whichset:
        for conversation in train_dev_split[sset]:
            all_br_len = []
            alltweets = [item for sublist in conversation['branches'] for item in sublist]
            uniqtweets = list(np.unique(alltweets))
            j = uniqtweets.index(conversation['source']['id_str'])
            del uniqtweets[j]   # now uniqtweets are replies only
            allrepliesfromfoldr = []
            for item in conversation['replies']:
                allrepliesfromfoldr.append(item['id_str'])
            if allrepliesfromfoldr != uniqtweets:
                # print "No correspondence between structure and replies"
                # print conversation['id']
                special.append(conversation['id'])

            for branch in conversation['branches']:
                branch_rep = []  # list of all tweets in the branch
                temp_rmd = []
                temp_label = []
                temp_id = []
                all_br_len.append(len(branch))
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
                    if sset != 'test':
                        label = tweet['label']
                        temp_label.append(convertlabel(label))  # convertlabel

                    temp_id.append(tweet['id_str'])
                    if tweet['used']:
                        # if tweet has been processed then take the representation
                        representation = tweet['representation']
                        temp_rmd.append(0)
                    else:
                        # if tweet is new then
                        # get tweet's representation
                        representation = tweet2features(model, tweet, i,
                                                        branch, conversation)
                        tweet['representation'] = representation
                        tweet['used'] = 1
                        temp_rmd.append(1)
                    branch_rep.append(representation)
                branch_list[sset].append(branch_rep)
                rmdoublemask_list[sset].append(temp_rmd)
                ID_list[sset].append(temp_id)
                if sset != 'test':
                    label_list[sset].append(temp_label)
            if max(all_br_len) > max_branch_len[sset]:
                max_branch_len[sset] = max(all_br_len)
    #%%
    # after that  transform those lists in numpy array,
    # get masks needed and saveto files

    branch_arrays = {}
    num_features = 314

    for sset in whichset:
        path_to_saved_data = './saved_data/fold%s/' % str(fold_num)
        path_to_save_sets = os.path.join(path_to_saved_data, sset)
        if not os.path.exists(path_to_save_sets):
            os.makedirs(path_to_save_sets)
        temp_list = []
        mask_list = []
        padlabel = []
        rmdoublemask = []
        ids = []
        for j, branch in enumerate(branch_list[sset]):
            # first put all tweets in branch to the temp array
            temp = np.zeros((max_branch_len[sset], num_features),
                            dtype=np.float32)
            temp_mask = np.zeros((max_branch_len[sset]), dtype=np.int32)
            temp_padlabel = np.zeros((max_branch_len[sset]), dtype=np.int32)
            temp_rmdoublemask = np.zeros((max_branch_len[sset]), dtype=np.int32)
            temp_ids = np.zeros((max_branch_len[sset]))
            temp_ids = [str(a) for a in temp_ids]
            for i, tweet in enumerate(branch):
                temp[i] = tweet
                temp_mask[i] = 1
                temp_rmdoublemask[i] = rmdoublemask_list[sset][j][i]
                temp_ids[i] = ID_list[sset][j][i]
                if sset != 'test':
                    temp_padlabel[i] = label_list[sset][j][i]
            temp_list.append(temp)
            mask_list.append(temp_mask)
            rmdoublemask.append(temp_rmdoublemask)
            ids.extend(temp_ids)
            if sset != 'test':
                padlabel.append(temp_padlabel)
        branch_arrays[sset] = np.asarray(temp_list)
        mask = np.asarray(mask_list)
        rmdoublemask = np.asarray(rmdoublemask)
        if sset != 'test':
            padlabel = np.asarray(padlabel)
        # save to files
        np.save(os.path.join(path_to_save_sets, 'rmdoublemask'), rmdoublemask)
        np.save(os.path.join(path_to_save_sets, 'mask'), mask)
        np.save(os.path.join(path_to_save_sets, 'branch_arrays'),
                branch_arrays[sset])
        with open(os.path.join(path_to_save_sets, 'ids.pkl'), 'wb') as f:
            pickle.dump(ids, f)
        if sset != 'test':
            np.save(os.path.join(path_to_save_sets, 'padlabel'), padlabel)

def Cross_validation_threads(train_dev_split):
    train_dev_splits = []
    one_split = {}
    dev_thread_number = len(train_dev_split['dev'])
    test_thread_number = len(train_dev_split['test'])
    np.random.seed(50)
    np.random.shuffle(train_dev_split['train'])
    for i in range(5):
        dev_threads = train_dev_split['train'][i * (dev_thread_number + test_thread_number) : i * (dev_thread_number + test_thread_number) + dev_thread_number]
        test_threads = train_dev_split['train'][i * (dev_thread_number + test_thread_number) + dev_thread_number : (i + 1) * (dev_thread_number + test_thread_number)]
        train_threads = [item for item in train_dev_split['train'] if item not in dev_threads and item not in test_threads]  + train_dev_split['dev'] + train_dev_split['test']
        one_split['train'] = train_threads
        one_split['dev'] = dev_threads
        one_split['test'] = test_threads
        train_dev_splits.append(one_split)
        one_split = {}
        #print "fold %s" % str(i)
        #print "train thread number:%s" % str(len(train_threads))
        #print "dev threads number:%s" % str(len(dev_threads))
        #print "test threads number:%s" % str(len(test_threads))
        #print "test threads ID:"
    return train_dev_splits

def save_test_labels(train_dev_splits, fold_num):
    train_dev_split = train_dev_splits[int(fold_num)]
    test_id_label = {}
    test_label_path = 'true_test_labels'
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)
    print ("save fold%s test labels" % str(fold_num))
    test_conv = train_dev_split['test']
    for conv in test_conv:
        src = conv['source']
        replies = conv["replies"]
        test_id_label[src['id_str']] = src['label']
        for reply in replies:
            test_id_label[reply['id_str']] = reply['label']
    with open(os.path.join(test_label_path,'fold%s.txt' % str(fold_num)), 'w+') as outfile:
        json.dump(test_id_label, outfile)

if __name__ == "__main__":

    # Read fold num for cross validation
    fold_num = sys.argv[1]
    print ("Preprocessing fold: %s" % str(fold_num))

    # Import NLTK data
    nltk_data_location = os.path.dirname(os.path.realpath(__file__))
    nltk.download('punkt', download_dir=nltk_data_location)

    train_dev_split = load_dataset()

    train_dev_splits = Cross_validation_threads(train_dev_split)

    #save true test dataset tweet_id:label
    save_test_labels(train_dev_splits, fold_num)

    # Import the data, preprocess it and store in the saved_data folder
    preprocess_data(train_dev_splits, fold_num)
    
    sys.stdout.flush()
