#coding = utf-8
import json
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def Load_data(output_path):
    with open(os.path.join(output_path, "dataset.txt"), "r") as outfile:
        line = outfile.readline()
        new_dataset = json.loads(line)
    outfile.close()

    with open(os.path.join(output_path, "first_predictions.txt"), "r") as outfile:
        line = outfile.readline()
        first_predictions = json.loads(line)
    outfile.close()

    with open(os.path.join(output_path, "second_predictions.txt"), "r") as outfile:
        line = outfile.readline()
        second_predictions = json.loads(line)
    outfile.close()

    return new_dataset, first_predictions, second_predictions

def output_result(dataset, first_predictions, second_predictions):
    #first predictions and labels only ["comment", "other"]
    first_prediction, first_labels = first_predictions[0], first_predictions[1]
    print ("*****************************")
    print ("First classification report: ")
    print ("*****************************")
    print (classification_report(first_labels, first_prediction, ["comment", "other"]))
    print (confusion_matrix(first_labels, first_prediction, labels=["comment", "other"]))

    #second predictions and labels only ["support", "deny", "query"]
    second_prediction, second_labels = second_predictions[0], second_predictions[1]
    print ("*****************************")
    print ("Second classification report: ")
    print ("*****************************")
    print (classification_report(second_labels, second_prediction, ["support", "deny", "query"]))
    print (confusion_matrix(second_labels, second_prediction, labels=["support", "deny", "query"]))

if __name__ == "__main__":
    new_dataset, first_predictions, second_predictions = Load_data("./output")
    output_result(new_dataset, first_predictions, second_predictions)
