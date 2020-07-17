import os
import ipdb
import json
import numpy
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from preprocessing import load_dataset, load_true_labels, split_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def convertlabeltostr(label):
    if label==0:
        return("support")
    elif label==1:
        return("comment")
    elif label==2:
        return("deny")
    elif label==3:
        return("query")
    else:
        print(label)  
  
def load_test_depth_pred_true(train_dev_splits, fold_num):

    # Read the predictions of the model
    submission_file = os.path.join("output_new", "prediction" + str(fold_num) + ".txt")
    submission = json.load(open(submission_file, 'r'))

    # And then the corresponding test data
    test_truevals = json.load(open('true_test_labels/fold%s.txt' % str(fold_num), 'r'))

    # Load the full dataset and get the list of test tweets and their properties
    train_dev_split = train_dev_splits[fold_num]
    alltestinfo = train_dev_split['test']

    alltestbranches = []
    # get all test branches out of it
    for indx, element in enumerate(alltestinfo):
        alltestbranches.extend(element['branches'])
    # loop over each tweet in testing set and find its depth to create id: depth dictionary
    depthinfo = {}
    for tweetid in submission.keys():
        for branch in alltestbranches:
            if tweetid in branch:
                depthinfo[tweetid] = branch.index(tweetid)

    return depthinfo, submission, test_truevals


def load_trials_data():

    trials_file = os.path.join("output", "trials.txt")

    if os.path.exists(trials_file):
        # Load the trials data
        with open(trials_file, 'rb') as f:
            trials = pickle.load(f)

        # We need to examine the best trial
        best_trial_id = trials.best_trial["tid"]
        best_trial_loss = trials.best_trial["result"]["loss"]
        dev_result_id = pickle.loads(trials.attachments["ATTACH::%d::ID" % best_trial_id])
        dev_result_predictions = pickle.loads(trials.attachments["ATTACH::%d::Predictions" % best_trial_id])

        # Change ID format from ints to strings
        strpred = [convertlabeltostr(s) for s in dev_result_predictions]

        # Transform to submission format and save
        dev_results_dict = dict(zip(dev_result_id, strpred))
        with open(os.path.join("output", "predictions_dev.txt"), "w") as outfile:
            json.dump(dev_results_dict, outfile)
    else:
        best_trial_id = None
        best_trial_loss = None
        dev_results_dict = None
        print "trials.txt is not available"

    return best_trial_id, best_trial_loss, dev_results_dict


def get_true_and_predicted_classes(true_tweet_classes, predicted_tweet_classes):

    # Sometimes this function may be called with empty predicted classes (e.g. if trials is not available).
    # In that case, return None for the true and predicted lists.
    if predicted_tweet_classes is None:
        true = None
        pred = None
        return true, pred

    true = []
    pred = []

    # Generate lists of true and predicted classes for all tweets in this set
    for k in true_tweet_classes.keys():
        true.append(true_tweet_classes[k])
        pred.append(predicted_tweet_classes[k])

    return true, pred


def calculate_results_at_each_depth(depthinfo, submission, test_truevals):

    # Group true labels and predictions according to their depth
    # depthinfo id: depth
    # submission id: prediction
    # test_truvals id: label

    depth_groups = {}
    depth_groups['0'] = []
    depth_groups['1'] = []
    depth_groups['2'] = []
    depth_groups['3'] = []
    depth_groups['4'] = []
    depth_groups['5'] = []
    depth_groups['6+'] = []


    # Find all keys in that depth group
    for tweetid, tweetdepth in depthinfo.iteritems():
        if tweetdepth == 0:
            depth_groups['0'].append(tweetid)
        elif tweetdepth == 1:
            depth_groups['1'].append(tweetid)
        elif tweetdepth == 2:
            depth_groups['2'].append(tweetid)
        elif tweetdepth == 3:
            depth_groups['3'].append(tweetid)
        elif tweetdepth == 4:
            depth_groups['4'].append(tweetid)
        elif tweetdepth == 5:
            depth_groups['5'].append(tweetid)
        elif tweetdepth >5:
            depth_groups['6+'].append(tweetid)

    # make a list

    depth_predictions = {}
    depth_predictions['0'] = []
    depth_predictions['1'] = []
    depth_predictions['2'] = []
    depth_predictions['3'] = []
    depth_predictions['4'] = []
    depth_predictions['5'] = []
    depth_predictions['6+'] = []

    depth_labels = {}
    depth_labels['0'] = []
    depth_labels['1'] = []
    depth_labels['2'] = []
    depth_labels['3'] = []
    depth_labels['4'] = []
    depth_labels['5'] = []
    depth_labels['6+'] = []

    depth_result = {}

    for depthgr in depth_groups.keys():
        depth_predictions[depthgr] = [submission[x] for x in depth_groups[depthgr]]
        depth_labels[depthgr] = [test_truevals[x] for x in depth_groups[depthgr]]

        _, _, mactest_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                             depth_predictions[depthgr],
                                                             average='macro')
        _, _, mictest_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                             depth_predictions[depthgr],
                                                             average='micro')
        _, _, test_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                          depth_predictions[depthgr])

        depth_result[depthgr] = [mactest_F, mictest_F, test_F]

    return depth_labels, depth_result


def print_table_three(true, pred, devtrue, devpred, best_trial_id, best_trial_loss):

    print "\n\n--- Table 3 ---"

    # Prepare headers for the version of Table 3 from the paper (we'll print some additional details too)
    table_three_headers = tuple(["", "Accuracy", "Macro-F"] + sorted(class_labels))
    results_headers = ("Precision", "Recall", "F-score", "Support")

    print "\nPart 1: Results on testing set"

    test_accuracy = accuracy_score(true, pred)
    print "\nAccuracy =", test_accuracy

    print "\nMacro-average:"
    macroavg_prfs = precision_recall_fscore_support(true, pred, average='macro')
    for lab, val in zip(results_headers, macroavg_prfs):
        if val is not None:
            print "%-12s%-12.3f" % (lab, val)
        else:
            print "%-12s%-12s" % (lab, "--")

    print "\nPer-class:"
    perclass_prfs = precision_recall_fscore_support(true, pred)
    print "%-12s%-12s%-12s%-12s%-12s" % tuple([""] + sorted(class_labels))
    for lab, vals in zip(results_headers, perclass_prfs):
        if lab is "Support":
            print "%-12s%-12i%-12i%-12i%-12i" % (lab, vals[0], vals[1], vals[2], vals[3])
        else:
            print "%-12s%-12.3f%-12.3f%-12.3f%-12.3f" % (lab, vals[0], vals[1], vals[2], vals[3])

    print "\nPart 2: Results on development set"

    if best_trial_id is not None:

        print "\nBest trial =", best_trial_id, "with loss", best_trial_loss

        #  Output is in the same format as the earlier part of Table 3
        dev_accuracy = accuracy_score(devtrue, devpred)
        print "Accuracy =", dev_accuracy

        print "\nMacro-average:"
        dev_macroavg_prfs = precision_recall_fscore_support(devtrue, devpred, average='macro')
        for lab, val in zip(results_headers, dev_macroavg_prfs):
            if val is not None:
                print "%-12s%-12.3f" % (lab, val)
            else:
                print "%-12s%-12s" % (lab, "--")

        print "\nPer-class:"
        dev_perclass_prfs = precision_recall_fscore_support(devtrue, devpred)
        print "%-12s%-12s%-12s%-12s%-12s" % tuple([""] + sorted(class_labels))
        for lab, vals in zip(results_headers, dev_perclass_prfs):
            if lab is "Support":
                print "%-12s%-12i%-12i%-12i%-12i" % (lab, vals[0], vals[1], vals[2], vals[3])
            else:
                print "%-12s%-12.3f%-12.3f%-12.3f%-12.3f" % (lab, vals[0], vals[1], vals[2], vals[3])

        print "\nAs presented in the paper:\n"
        print "%-12s%-12s%-12s%-12s%-12s%-12s%-12s" % table_three_headers
        print "%-12s%-12.3f%-12.3f%-12.3f%-12.3f%-12.3f%-12.3f" % (("Development", dev_accuracy, dev_macroavg_prfs[2]) + tuple(dev_perclass_prfs[2]))
        print "%-12s%-12.3f%-12.3f%-12.3f%-12.3f%-12.3f%-12.3f" % (("Testing", test_accuracy, macroavg_prfs[2]) + tuple(perclass_prfs[2]))

    # trials.txt is generated by the parameter search, and so won't be generated by outer.py unless requested
    # (substantial compute resources needed). So we may not be able to generate this part of the table.
    else:

        print "\nAs presented in the paper:\n"
        print "%-12s%-12s%-12s%-12s%-12s%-12s%-12s" % table_three_headers
        print "%-12s%-12.3f%-12.3f%-12.3f%-12.3f%-12.3f%-12.3f" % (("Testing", test_accuracy, macroavg_prfs[2]) + tuple(perclass_prfs[2]))
        print "\nCould not find trials.txt; unable to generate results for development set in Table 3.\n"


def print_extra_details(best_trial_id):

    trials = pickle.load(open(os.path.join("output", "trials.txt"), "rb"))

    # Print out the best combination of hyperparameters
    print "\n--- New Table ---\n"
    print "The best combination of hyperparameters, found in trial " + str(best_trial_id) + ", was:"
    for param, param_value in trials.best_trial["result"]["Params"].iteritems():
        print "\t", param, "=", param_value

    # Let's examine the loss function at each iteration of the hyperparameter tuning process
    print "\n--- New Figure ---"

    # Extract the loss values from the full list of results, and calculate the running minimum value
    loss = numpy.asarray([r["loss"] for r in trials.results])
    running_min_loss = numpy.minimum.accumulate(loss)
    lowest_loss = loss[best_trial_id]
    all_best_ids = numpy.where(loss == lowest_loss)[0]

    # Plot the loss and running loss values against the iteration number, and save to the output folder
    plt.plot(range(0, len(loss)), loss, label="loss")
    plt.plot(range(0, len(running_min_loss)), running_min_loss, label="running min(loss)")
    plt.plot(best_trial_id, lowest_loss, "ro", label="min(loss)")
    if len(all_best_ids) > 1:
        plt.plot(all_best_ids, lowest_loss*numpy.ones(all_best_ids.shape), "rx", label="repeated min(loss)")
    plt.legend()
    plt.title("Hyperparameter optimisation")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(os.path.join("output", "hyperparameter_loss_values.pdf"))

    # Give details of other hyperparameter combinations that also achieved this loss
    if len(all_best_ids) > 1:
        print "\nWARNING: multiple hyperparameter combinations achieved the same lowest loss value as trial", best_trial_id
        print "ID               ",
        for id in all_best_ids:
            print "%-17d" % id,
        print ""
        for param in trials.results[all_best_ids[0]]["Params"]:
            print "%-17s" % param,
            for id in all_best_ids:
                print "%-17.5g" % trials.results[id]["Params"][param],
            print ""

    print "\nFigure showing hyperparameter optimisation progress can be found in the output folder.\n"


def print_table_four(depth_labels, depth_result):

    print "\n\n--- Table 4 ---"
    print "\nNumber of tweets per depth and performance at each of the depths\n"

    # Print the column headers
    table_four_headers = ("Depth", "# tweets", "# Support", "# Deny", "# Query", "# Comment", "Accuracy", "MacroF") + class_labels
    for col in table_four_headers:
        print "%-11s" % col,
    print ""

    #  Print results in depth level order
    for depth in sorted(depth_result):

        # Work out which class the accuracy values refer to (precision_recall_fscore_support() outputs values in the
        # sorted order of the unique classes of tweets at that depth)
        depth_class_accuracy = depth_result[depth][2]
        depth_class_labels = sorted(set(depth_labels[depth]))

        # Print the depth and classes of tweets at that depth
        print "%-12s%-11i" % (depth, len(depth_labels[depth])),
        for lab in class_labels:
            print "%-11i" % depth_labels[depth].count(lab.lower()),

        # Print the accuracy, macro-F and class-specific performance at each depth
        print "%-12.3f%-11.3f" % \
              (depth_result[depth][1], depth_result[depth][0]),
        for lab in class_labels:
            if lab.lower() in depth_class_labels:
                class_ind = depth_class_labels.index(lab.lower())
                print "%-11.3f" % depth_class_accuracy[class_ind],
            else:
                print "%-11.3f" % 0.0,
        print ""


def print_table_five(true, pred):

    print "\n\n--- Table 5 ---"
    print "\nConfusion matrix\n"

    # Generate the confusion matrix and the list of labels (as above, in sorted class order as long as each class
    # appears once, which they all do).
    conf_mat = confusion_matrix(true, pred)
    class_labels_mat = ("Lab \\ Pred",) + tuple(sorted(class_labels))

    # Print the header and then the confusion matrix
    print "%-12s%-12s%-12s%-12s%-12s" % class_labels_mat
    for lab, conf_row in zip(sorted(class_labels), conf_mat):
        row = (lab,) + tuple(conf_row)
        print "%-12s%-12i%-12i%-12i%-12i" % row


if __name__ == "__main__":

    # First load the full set of conversations.
    # Then calculate the depth and extract the true and predicted labels for the test set specifically.
    allconv = load_dataset()
    train_dev_splits = split_dataset(allconv)

    tweet_depth_list = []
    test_predicted_labels_list = []
    test_labels_list = []

    for fold_num in range(5):
        tweet_depth, test_predicted_labels, test_labels = load_test_depth_pred_true(train_dev_splits, fold_num)
        tweet_depth_list.append(tweet_depth)
        test_predicted_labels_list.append(test_predicted_labels)
        test_labels_list.append(test_labels)

    # If it is present, load data from trials file and format in the same way as the submitted files
    # (return None if the trials file is not available)
    #dev_labels = load_true_labels("dev")
    dev_labels = None
    best_trial, best_loss, dev_predicted_labels = load_trials_data()

    # Define some useful labels for table rows/columns
    class_labels = ("Support", "Deny", "Query", "Comment")

    # Analyse the results separately at each depth
    for i in range(5):
        level_for_each_depth, results_for_each_depth = \
            calculate_results_at_each_depth(tweet_depth_list[i], test_predicted_labels_list[i], test_labels_list[i])

        # Get lists of the true and predicted classes for the test and, if possible, development sets
        #ipdb.set_trace()
        true_labels_test, predicted_labels_test = get_true_and_predicted_classes(test_labels_list[i], test_predicted_labels_list[i])
        true_labels_dev, predicted_labels_dev = get_true_and_predicted_classes(dev_labels, dev_predicted_labels)


        # Print the tables
        print "***************************"
        print "********  Fold %s   ********" % str(i)
        print "***************************"
        print_table_four(level_for_each_depth, results_for_each_depth)
        print_table_five(true_labels_test, predicted_labels_test)
        print_table_three(true_labels_test, predicted_labels_test,
                          true_labels_dev, predicted_labels_dev, best_trial, best_loss)

    # If the trials file is available, output more details of the best hyperparameter combinations and prepare a figure
    # showing the loss during the hyperparameter choice process
    if best_trial is not None:
        print_extra_details(best_trial)
