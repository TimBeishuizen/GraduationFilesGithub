import csv
import matplotlib.pyplot as PLT
import numpy as np

val = []
test = []
feat = []

used_test = "MI"
#used_test = "T"

#used_data = "RSCTC"
#used_data = "Psoriasis"
#used_data = "Arcene"
used_data = "MO"


with open(used_data + '_' + used_test + '_val_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        val.append(row)


with open(used_data + '_' + used_test + '_test_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            test.append(row)

with open(used_data + '_' + used_test + '_feat_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            feat.append(row)


feat = np.array(feat, dtype=int)
val = np.array(val, dtype=float)
test = np.array(test, dtype=float)


PLT.plot(feat[1, 1:], val[1, 1:], 'b-', feat[1, 1:], test[1, 1:], 'b--', feat[2, 1:], val[2, 1:], 'r-', feat[2, 1:], test[2, 1:], 'r--',
         feat[3, 1:], val[3, 1:], 'g-', feat[3, 1:], test[3, 1:], 'g--', feat[0, 1:], val[0, 1:], 'c-', feat[0, 1:], test[0, 1:], 'c--',
         feat[4, 1:], val[4, 1:], 'y-', feat[4, 1:], test[4, 1:], 'y--')
PLT.legend(["Decision tree validation score", "Decision tree test score", "Nearest Neighbour validation score",
           "Nearest Neighbour test score", "SVM validation score", "SVM test score",
            "Logistic regression validation score", "Logistic regression test score", "Naive Bayes validation score",
            "Naive Bayes test score"])
if used_test == 'T':
    PLT.title('Scores for features selected by T-test for dataset Micro Organisms')
elif used_test == 'MI':
    PLT.title('Scores for features selected by Mutual information for dataset Micro Organisms')

PLT.xlabel('Number of features')
PLT.ylabel('Scores for validation (not interrupted) and testing (dotted)')


PLT.show()