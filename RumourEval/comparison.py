#coding=utf-8
'''compare the performance of BERT and branch-LSTM on different levels of dataset'''
import re
import matplotlib.pyplot as plt

def Load_accuracy(result_file):
    f = open(result_file, 'r')
    depth = 0
    in_table = False
    accuracy_list = []
    for line in f.readlines():
        if re.match(r'Depth', line):
            in_table = True
            continue

        if in_table:
            #print (line)
            accuracy = re.findall(r'(?:\d+(?:\+)?\s+){6}([\.\d]+)', line)
            #print (accuracy)
            accuracy_list.append(accuracy[0])
            depth = depth + 1
            if depth == 7:
                break
        else:
            continue
    return accuracy_list

def Load_predictions(prediction_file):
    prediction_list = []
    return prediction_list

def Plot_line_chart(list1, list2):
    plt.xlabel("tweet depth")
    plt.ylabel("accuracy")
    plt.title("accuracy on different depth comparison")
    #plt.figure()
    plt.axis([0, len(list1), 0, 1])
    plt.plot(range(len(list1)), list1)
    #plt.plot(list2, range(len(list2)))
    plt.savefig("./line_chart.png", dpi=400)

def Plot_confusion_matrix(list1, list2):
    plt.save_fig()

if __name__ == "__main__":
    #load performance of BERT and branch-LSTM: accuracy, marco F and per class on every level post
    branchLSTM_result = "./branchLSTM/output/tables.txt"
    BERT_result = "./BERT_baseline/output/tables.txt"

    branchLSTM_predictions = './branchLSTM/output/predictions.txt'
    BERT_predictions = './BERT_baseline/output/predictions.txt'

    branchLSTM_accuracy = Load_accuracy(branchLSTM_result)
    BERT_accuracy = Load_accuracy(BERT_result)
    print (branchLSTM_accuracy)
    print (BERT_accuracy)

    #branchLSTM_prediction_list = Load_predictions(branchLSTM_predictions)
    #BERT_prediction_list = Load_predictions(BERT_predictions)

    Plot_line_chart(branchLSTM_accuracy, BERT_accuracy)
    #Plot_confusion_matrix(branchLSTM_prediciton_list, BERT_prediction_list)
