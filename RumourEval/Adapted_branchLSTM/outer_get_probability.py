#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Contains the following functions: 
   
parameter search - defines parameter space,performs parameter search using 
                      objective_train_model and hyperopt TPE search.
convertlabeltostr - converts int label to str
eval - passes parameters to eval_train_model, does results postprocessing 
          to fit with scorer.py and saves results
main - brings all together, controls command line arguments

Run outer.py

python outer.py

outer.py has the following options:

python outer.py --search=True --ntrials=10 --params="output/bestparams.txt"

--search  - boolean, controls whether parameter search should be performed
--ntrials - if --search is True then this controls how many different 
            parameter combinations should be assessed
--params - specifies filepath to file with parameters if --search is false

-h, --help - explains the command line 


If performing parameter search, then execution will take long time 
depending on number of trials, size and number of layers in parameter space. 
Use of GPU is highly recommended. 

If running with default parametes then search won't be performed and 
parameters will be used from 'output/bestparams.txt'

"""
import sys
import timeit
import os
import pickle
import json
#os.environ["THEANO_FLAGS"]="floatX=float32"
#os.environ["THEANO_FLAGS"]="floatX=float32,dnn.enabled=False,cxx=icpc,
#                            device=gpu0,nvcc.compiler_bindir=icpc,
#                            gcc.cxxflags=-march=native"
import numpy
from hyperopt import fmin, tpe, hp, Trials
from training import objective_train_model
from predict_get_probability import eval_train_model
from optparse import OptionParser

print ("Load Libraries done!\n")

def parameter_search(ntrials, hyperopt_seed, is_test=False):
    start = timeit.default_timer()
    trials = Trials()

    if is_test:
        # Unrealistic values of parameters; for testing use only
        search_space= { 'num_dense_layers': hp.choice('nlayers',[1, 2, 3, 4]),
                        'num_dense_units': hp.choice('num_dense', [2, 3, 4, 5]),
                        'num_epochs': hp.choice('num_epochs',  [3, 5, 7, 10]),
                        'num_lstm_units': hp.choice('num_lstm_units',  [2, 3]),
                        'num_lstm_layers': hp.choice('num_lstm_layers', [1, 2]),
                        'learn_rate': hp.choice('learn_rate',[1e-4, 3e-4, 1e-3]),
                        'mb_size': hp.choice('mb_size', [32, 64, 100, 120]),
                        'l2reg': hp.choice('l2reg', [0.0, 1e-4, 3e-4, 1e-3]),
                        'rng_seed': hp.choice('rng_seed', [364])
    }
    else:
        # Parameter values tested for SemEval-2017 Task 8
        search_space= { 'num_dense_layers': hp.choice('nlayers',[1,2,3,4]),
                        'num_dense_units': hp.choice('num_dense', [100, 200, 300,
                                                                   400, 500]),
                        'num_epochs': hp.choice('num_epochs',  [30, 50, 70, 100]),
                        'num_lstm_units': hp.choice('num_lstm_units',  [100, 200,
                                                                        300]),
                        'num_lstm_layers': hp.choice('num_lstm_layers', [1,2]),
                        'learn_rate': hp.choice('learn_rate',[1e-4, 3e-4, 1e-3]),
                        'mb_size': hp.choice('mb_size', [32, 64, 100, 120]),
                        'l2reg': hp.choice('l2reg', [0.0, 1e-4, 3e-4, 1e-3]),
                        'rng_seed': hp.choice('rng_seed', [364])
        }

    # Find the best set of hyperparameters (with TPE algorithm)
    best = fmin(objective_train_model,
                space=search_space,
                algo=tpe.suggest,
                max_evals=ntrials,
                trials=trials,
                rstate=numpy.random.RandomState(hyperopt_seed))
    
    params = trials.best_trial['result']['Params']

    print ("\nBest hyperparameters are:")
    for hp_name, hp_value in params.items():
        print "%-20s%-.5g" % (hp_name, hp_value)
        
    out_path = 'output'
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    
    f = open(os.path.join(out_path, 'trials.txt'), "w+")
    pickle.dump(trials, f)
    f.close()
    
    f = open(os.path.join(out_path, 'bestparams.txt'), "w+")
    pickle.dump(params, f)
    f.close()
    
    print "\nSaved trials and optimal hyperparameters to trials.txt and bestparams.txt"
    
    stop = timeit.default_timer()
    print "Total time: ", stop - start, "\n"
    
    return params

def eval(params, fold_num):
    start = timeit.default_timer()
    eval_train_model(params, fold_num)
    stop = timeit.default_timer()
    print ("Time: ",stop - start)

def main(fold_num):
    parser = OptionParser()
    parser.add_option(
            '--search', dest='psearch', default=False,
            help='Whether parameter search should be done: default=%default')
    parser.add_option('--ntrials', dest='ntrials', default=10,
                      help='Number of trials: default=%default')
    parser.add_option('--hseed', dest='hseed', default=None,
            help="Value of the rng seed passed to hyperopt's fmin function: default=%default")
    parser.add_option(
            '--params', dest='params_file', default='output/bestparams_semeval2017.txt',
            help='Location of parameter file: default=%default')
    parser.add_option(
            '--test', dest='is_test', default=False,
            help='Run with test parameters: default=%default')
    print ("parameters setting done!\n")

    
    
    (options, args) = parser.parse_args()
    psearch = options.psearch
    ntrials = int(options.ntrials)
    hyperopt_seed = options.hseed
    if hyperopt_seed is not None:
        hyperopt_seed = int(hyperopt_seed)
    is_test = options.is_test
    params_file = options.params_file
    print ("search or not search start:\n")
    
    if psearch:
        if is_test:
            print '\nStarting parameter search using test parameters...\n'
        else:
            print '\nStarting parameter search...\n'
        params = parameter_search(ntrials, hyperopt_seed, is_test)
        print(params)
        eval(params. fold_num)
    else:
        with open(params_file, 'rb') as f:
            print '\nLoading best set of model parameters from ', params_file, '...\n'
            params = pickle.load(f)
        print (params)
        eval(params, fold_num)
        
#%%


if __name__ == '__main__':
    fold_num = sys.argv[1]
    print ("Fold %s main function:\n" % str(fold_num))
    main(fold_num)

    sys.stdout.flush()
