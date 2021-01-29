# -*- coding: utf-8 -*-
#environemnt : ktrain
#usage: CUDA_VISIBLE_DEVICES=x python Classify.py > log
#predictions saved in ./output/predictions.txt
import sys
import os
import timeit
import json
import ktrain
from ktrain import text
from sklearn.metrics import classification_report
from preprocessing import load_dataset, Cross_validation_threads, save_test_labels

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
    start = timeit.default_timer()
    out_path = './output'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    result_dictionary = dict(zip(id_list, predictions))

    with open(os.path.join(out_path,'prediction_fold%s.txt' % str(fold_num)), 'w+') as outfile:
        json.dump(result_dictionary, outfile)
    print ("saved result and predictions")
    stop = timeit.default_timer()
    print ("Time: ",stop - start)

def Classify(dataset, fold_num):
    #dataset[dataset_name] = [tweet_list, label_list, id_list]

    categories = ['support', 'query', 'deny', 'comment']
    #print ("categories:", categories)
    #categories2num = {'support':0, 'query':1, 'deny':2, 'comment':3}

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

    predictions = []
    for tweet in test_texts:
        predictions.append(p.predict(tweet))

    tweet_IDs = dataset['test'][2]

    save_predictions(tweet_IDs, predictions, fold_num)

    #print (classification_report(test_labels, p_test, list(np.unique(test_labels))))

if __name__ == "__main__":
    fold_num = sys.argv[1]
    print ("Fold %s classification!" % str(fold_num))

    # Calculate the depth and extract the true and predicted labels for the test set specifically.
    train_dev_split = load_dataset()
    train_dev_splits = Cross_validation_threads(train_dev_split)

    #save true test dataset tweet_id:label
    save_test_labels(train_dev_splits, fold_num)

    dataset = Extract_dataset(train_dev_splits[int(fold_num)])
    #print (dataset.keys())
    Classify(dataset, fold_num)
