#coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt

Depth = ['0', '1', '2', '3', '4', '5', '6+']
#models = ['BranchLSTM', 'DistilBERT', 'SVM', 'CRF', 'CRF+SVM', 'CRF+CRF']
#models = ['SVM', 'CRF', 'CRF+SVM', 'CRF+CRF']
#models = ['BranchLSTM', 'SVM']
models = ['DistilBERT', 'SVM', 'BranchLSTM']

LSTM_macro_F_depth = [0.6220, 0.4136, 0.2534, 0.2918, 0.2594, 0.4660, 0.3928]
BERT_macro_F_depth = [0.4698, 0.4762, 0.3312, 0.3622, 0.3198, 0.4602, 0.4832]

SVM_macro_F_depth = [0.4712, 0.4650, 0.3488, 0.3574, 0.3122, 0.5422, 0.4598]
CRF_macro_F_depth = [0.5026, 0.4586, 0.3536, 0.3808, 0.3292, 0.4796, 0.4594]

CRF_SVM_F_depth = [0.5046, 0.4614, 0.3586, 0.3520, 0.3232, 0.5386, 0.4544]
CRF_CRF_F_depth = [0.5026, 0.4588, 0.3352, 0.3808, 0.3292, 0.4796, 0.4594]

#Data = {'BranchLSTM':LSTM_macro_F_depth, 'DistilBERT':BERT_macro_F_depth,
#        'SVM':SVM_macro_F_depth, 'CRF':CRF_macro_F_depth,
#        'CRF+SVM':CRF_SVM_F_depth, 'CRF+CRF':CRF_CRF_F_depth}

Data = {'SVM':SVM_macro_F_depth,
        'DistilBERT':BERT_macro_F_depth,
        'BranchLSTM':LSTM_macro_F_depth}

DF = pd.DataFrame(data=Data, columns=models, index=Depth)
print (DF)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

ax = DF.plot(marker='o')

for line, name in zip(ax.lines, DF.columns):
    y = line.get_ydata()[-1]
    ax.annotate(name, xy=(1,y), xytext = (1,0), color=line.get_color(), xycoords = ax.get_yaxis_transform(), textcoords='offset points', size = 14, va = 'center')
    #ax.annotate(name, xy=(1,y), color=line.get_color())

ax.set_xlabel('Depth')
ax.set_title('Macro-F1 by depth')
#plt.ylabel()
plt.show()

