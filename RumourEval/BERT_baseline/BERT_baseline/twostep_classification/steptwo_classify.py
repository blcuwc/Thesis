# -*- coding: utf-8 -*-
#environment : ktrain
# usage : CUDA_VISIBLE_DEVICES=x python steptwo_classify.py > steptwo_log
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

def Load_dataset(output_path):
    with open(os.path.join(output_path, "dataset.txt"), "r") as outfile:
        line = outfile.readline()
        new_dataset = json.loads(line)
    outfile.close()

    with open(os.path.join(output_path, "rest_test_dataset.txt"), "r") as outfile:
        line = outfile.readline()
        rest_test_dataset = json.loads(line)
    outfile.close()

    with open(os.path.join(output_path, "first_predictions.txt"), "r") as outfile:
        line = outfile.readline()
        first_predictions = json.loads(line)
    outfile.close()

    return new_dataset, rest_test_dataset, first_predictions

def Second_step_classify(dataset, rest_test_dataset, text):
    categories = ["support", "query", "deny"]
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = rest_test_dataset['text']
    y_test = rest_test_dataset['labels']

    #use all other three classes data to train and validate classifier
    for i in range(len(dataset['train'][1])):
        if dataset['train'][1][i] != "comment":
            x_train.append(dataset['train'][0][i])
            y_train.append(dataset['train'][1][i])

    for i in range(len(dataset['dev'][1])):
        if dataset['dev'][1][i] != "comment":
            x_dev.append(dataset['dev'][0][i])
            y_dev.append(dataset['dev'][1][i])
    #print ("x_train: ", x_train)
    print ("y_train: ", y_train)
    #print ("x_dev: ", x_dev)
    print ("y_dev: ", y_dev)

    trn, val, preproc = text.texts_from_array(x_train = x_train, y_train = y_train,
                                          x_test = x_dev, y_test = y_dev,
                                          class_names = categories,
                                          preprocess_mode = 'distilbert',
                                          maxlen = 350)
    print ("second step ktrain preprocess finished!")
    #text.print_text_classifiers()

    model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
    print ("second classifier built finished!")

    #learning rate and epochs
    learner.fit_onecycle(3e-5, 4)
    print ("second classifier training finished!")

    p = ktrain.get_predictor(model, preproc)

    predictions = []
    for text in x_test:
        predictions.append(p.predict(text))
    #print ("second predictions: ", predictions)
    #print ("second labels: ", y_test)
    print ("*****************************")
    print ("Second classification report: ")
    print ("*****************************")
    print (classification_report(y_test, predictions, categories))

    with open(os.path.join("./output", "second_predictions.txt"), "w") as outfile:
        json.dump([predictions, y_test], outfile)
    outfile.close()

if __name__ == "__main__":
    new_dataset, rest_test_dataset, first_predictions = Load_dataset("./output")
    print ("new_dataset: ", new_dataset.keys())
    print ("rest_test_dataset: ", rest_test_dataset.keys())
    print ("first_predictions: ", first_predictions[0])
    Second_step_classify(new_dataset, rest_test_dataset, text)
