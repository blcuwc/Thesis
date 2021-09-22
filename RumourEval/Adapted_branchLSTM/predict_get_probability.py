#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Contains the following functions: 
   
   eval_train_model - re-trains model on train+dev set and 
                      evaluates on test set
"""
import numpy
import theano
import theano.tensor as T
import lasagne
import os
import re
import pickle
from hyperopt import STATUS_OK
from training import build_nn,iterate_minibatches
#theano.config.floatX = 'float32'
#theano.config.warn_float64 = 'raise'
#%%


def eval_train_model(params, fold_num):
    print "Fold %s retrain model on train+dev set and evaluate on testing set" % str(fold_num)
    # Initialise parameters 
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int(params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']
    rng_seed = params['rng_seed']
#%%
    # Load data
    path = 'saved_data/fold%s/' % str(fold_num)
    train_arrays_file = open(os.path.join(path, 'train/branch_arrays.npy'))
    train_arrays_max_len = re.findall(r'\d+', train_arrays_file.readline())[-2]
    print "train branch arrays max len: %s" % train_arrays_max_len
    dev_arrays_file = open(os.path.join(path, 'dev/branch_arrays.npy'))
    dev_arrays_max_len = re.findall(r'\d+', dev_arrays_file.readline())[-2]
    print "dev branch arrays max len: %s" % dev_arrays_max_len

    train_brancharray = numpy.load(os.path.join(path, 'train/branch_arrays.npy'))
    num_features = numpy.shape(train_brancharray)[-1]
    train_mask = numpy.load(os.path.join(path,
                                         'train/mask.npy')).astype(numpy.int16)
    train_label = numpy.load(os.path.join(path, 'train/padlabel.npy'))
    
    train_rmdoublemask = numpy.load(
                            os.path.join(
                                path,
                                'train/rmdoublemask.npy')).astype(numpy.int16)
    train_rmdoublemask = train_rmdoublemask.flatten()
#%%
    numpy.random.seed(rng_seed)
    rng_inst = numpy.random.RandomState(rng_seed)
    lasagne.random.set_rng(rng_inst)
    input_var = T.ftensor3('inputs')
    mask = T.wmatrix('mask')
    target_var = T.ivector('targets')
    rmdoublesmask = T.wvector('rmdoublemask')
    # Build network
    network = build_nn(input_var, mask, num_features,
                       num_lstm_layers=num_lstm_layers,
                       num_lstm_units=num_lstm_units,
                       num_dense_layers=num_dense_layers,
                       num_dense_units=num_dense_units)
    # This function returns the values of the parameters of all
    # layers below one or more given Layer instances,
    # including the layer(s) itself.

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss*rmdoublesmask
    loss = lasagne.objectives.aggregate(loss, mask.flatten())
    # regularisation
    l2_penalty = l2reg * lasagne.regularization.regularize_network_params(
                                network,
                                lasagne.regularization.l2)
    loss = loss + l2_penalty

    # We could add some weight decay as well here, see lasagne.regularization.
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step.
    parameters = lasagne.layers.get_all_params(network, trainable=True)
    my_updates = lasagne.updates.adam(loss, parameters,
                                      learning_rate=learn_rate)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function(inputs=[input_var, mask,
                                       rmdoublesmask, target_var],
                               outputs=loss,
                               updates=my_updates,
                               on_unused_input='warn')
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, mask], test_prediction,
                             on_unused_input='warn')
#%%
    # READ THE DATA
    with open(os.path.join(path,'train/ids.pkl'), 'rb') as handle:
        train_ids_padarray = pickle.load(handle)
    
    dev_brancharray = numpy.load(os.path.join(path, 'dev/branch_arrays.npy'))
    dev_mask = numpy.load(
               os.path.join(
                   path,
                   'dev/mask.npy')).astype(numpy.int16)
    dev_label = numpy.load(os.path.join(path, 'dev/padlabel.npy'))

    dev_rmdoublemask = numpy.load(
                       os.path.join(
                        path,
                        'dev/rmdoublemask.npy')).astype(numpy.int16).flatten()

    with open(os.path.join(path,'dev/ids.pkl'), 'rb') as handle:
        dev_ids_padarray = pickle.load(handle)
    
    test_brancharray = numpy.load(os.path.join(path, 'test/branch_arrays.npy'))
    test_mask = numpy.load(
                os.path.join(
                    path,
                    'test/mask.npy')).astype(numpy.int16)

    test_rmdoublemask = numpy.load(
                os.path.join(path,
                             'test/rmdoublemask.npy')).astype(
                                                       numpy.int16).flatten()
                
    with open(os.path.join(path,'test/ids.pkl'), 'rb') as handle:
        test_ids_padarray = pickle.load(handle)

#%%
    #start training loop
    # We iterate over epochs:
    for epoch in range(num_epochs):
        #print("Epoch {} ".format(epoch))
        train_err = 0
        # In each epoch, we do a full pass over the training data:
        for batch in iterate_minibatches(train_brancharray, train_mask,
                                         train_rmdoublemask,
                                         train_label, mb_size,
                                         max_seq_len=int(train_arrays_max_len), shuffle=False):
                inputs, mask, rmdmask, targets = batch
                train_err += train_fn(inputs, mask,
                                      rmdmask, targets)
        for batch in iterate_minibatches(dev_brancharray, dev_mask,
                                         dev_rmdoublemask,
                                         dev_label, mb_size,
                                         max_seq_len=int(dev_arrays_max_len), shuffle=False):
                inputs, mask, rmdmask, targets = batch
                train_err += train_fn(inputs, mask,
                                      rmdmask, targets)
    # And a full pass over the test data:
    train_ypred = val_fn(train_brancharray, train_mask)
    dev_ypred = val_fn(dev_brancharray, dev_mask)
    test_ypred = val_fn(test_brancharray, test_mask)

    #Take mask into account
    train_mask_flatten = train_mask.flatten()
    dev_mask_flatten = dev_mask.flatten()
    test_mask_flatten = test_mask.flatten()

    clip_train_ids = [o for o, m in zip(train_ids_padarray, train_mask_flatten) if m == 1]
    clip_train_probabilities = [o for o, m in zip(train_ypred, train_mask_flatten)
                           if m == 1]

    clip_dev_ids = [o for o, m in zip(dev_ids_padarray, dev_mask_flatten) if m == 1]
    clip_dev_probabilities = [o for o, m in zip(dev_ypred, dev_mask_flatten)
                           if m == 1]

    clip_test_ids = [o for o, m in zip(test_ids_padarray, test_mask_flatten) if m == 1]
    clip_test_probabilities = [o for o, m in zip(test_ypred, test_mask_flatten)
                           if m == 1]

    # remove repeating train instances
    uniqtwid_train, uindices_train = numpy.unique(clip_train_ids, return_index=True)
    uniq_train_probabilities = [clip_train_probabilities[i] for i in uindices_train]
    uniq_train_id = [clip_train_ids[i] for i in uindices_train]
    train_output = {'Probabilities': uniq_train_probabilities,'ID': uniq_train_id}
    train_probability_file = open(os.path.join(path, 'train/probabilities.txt'), 'wb')
    pickle.dump(train_output, train_probability_file, -1)

    # remove repeating dev instances
    uniqtwid_dev, uindices_dev = numpy.unique(clip_dev_ids, return_index=True)
    uniq_dev_probabilities = [clip_dev_probabilities[i] for i in uindices_dev]
    uniq_dev_id = [clip_dev_ids[i] for i in uindices_dev]
    dev_output = {'Probabilities': uniq_dev_probabilities,'ID': uniq_dev_id}
    dev_probability_file = open(os.path.join(path, 'dev/probabilities.txt'), 'wb')
    pickle.dump(dev_output, dev_probability_file, -1)

    # remove repeating test instances
    uniqtwid_test, uindices_test = numpy.unique(clip_test_ids, return_index=True)
    uniq_test_probabilities = [clip_test_probabilities[i] for i in uindices_test]
    uniq_test_id = [clip_test_ids[i] for i in uindices_test]
    test_output = {'Probabilities': uniq_test_probabilities,'ID': uniq_test_id}
    test_probability_file = open(os.path.join(path, 'test/probabilities.txt'), 'wb')
    pickle.dump(test_output, test_probability_file, -1)

