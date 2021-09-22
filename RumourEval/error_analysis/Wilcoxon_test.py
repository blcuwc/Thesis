# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
# seed the random number generator
seed(1)
# generate two independent samples
#data1 = 5 * randn(100) + 50
#data2 = 5 * randn(100) + 51
BERT_acc1 = [0.712, 0.728, 0.738]
BERT_acc2 = [0.765, 0.798, 0.691]
XLNet_acc1 = [0.747, 0.746, 0.772]
XLNet_acc2 = [0.766, 0.808, 0.721]

BranchLSTM = [0.704, 0.525, 0.459, 0.457, 0.704, 0.704, 0.704]
DistilBERT = [0.688, 0.530, 0.497, 0.504, 0.688, 0.688, 0.688]

# compare samples
stat1, p1 = wilcoxon(x = BERT_acc1, y = XLNet_acc1)
stat2, p2 = wilcoxon(x = BERT_acc2, y = XLNet_acc2)
stat3, p3 = wilcoxon(x = BranchLSTM, y = DistilBERT)


print('polarity : Statistics=%.3f, p=%.3f' % (stat1, p1))
print('factuality : Statistics=%.3f, p=%.3f' % (stat2, p2))
print('Statistics=%.3f, p=%.3f' % (stat3, p3))

# interpret
alpha = 0.05
if p1 > alpha:
    print('fail to reject H0')
else:
    print('reject H0')

if p2 > alpha:
    print('fail to reject H0')
else:
    print('reject H0')

if p3 > alpha:
    print('fail to reject H0')
else:
    print('reject H0')
