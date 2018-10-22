import csv
import matplotlib.pyplot as PLT
import numpy as np

val = []
test = []
feat = []
prec = []
rec = []
F1 = []

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

with open(used_data + '_' + used_test + '_prec_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        prec.append(row)

with open(used_data + '_' + used_test + '_rec_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            rec.append(row)

with open(used_data + '_' + used_test + '_F1_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            F1.append(row)


feat = np.array(feat, dtype=int)
val = np.array(val, dtype=float)
test = np.array(test, dtype=float)
prec = np.array(prec, dtype=float)
rec = np.array(rec, dtype=float)

F1 = np.array(F1, dtype=float)

plot_list = val

# PLT.plot(feat[1, 1:], plot_list[1, 1:], 'b-', feat[2, 1:], plot_list[2, 1:], 'r-', feat[3, 1:], plot_list[3, 1:], 'g-',
#          feat[0, 1:], plot_list[0, 1:], 'c-', feat[4, 1:], plot_list[4, 1:], 'y-')

PLT.plot(feat[1, 1:], val[1, 1:], 'b-', feat[1, 1:], test[1, 1:], 'b--', feat[2, 1:], val[2, 1:], 'r-', feat[2, 1:], test[2, 1:], 'r--',
         feat[3, 1:], val[3, 1:], 'g-', feat[3, 1:], test[3, 1:], 'g--', feat[0, 1:], val[0, 1:], 'c-', feat[0, 1:], test[0, 1:], 'c--',
         feat[4, 1:], val[4, 1:], 'y-', feat[4, 1:], test[4, 1:], 'y--')
#
PLT.legend(["Decision tree", "Nearest Neighbour",
           "SVM", "Logistic regression", "Naive Bayes"])
if used_test == 'T':
    PLT.title('Scores for features selected by T-test for dataset Micro Organisms')
elif used_test == 'MI':
    PLT.title('Scores for features selected by Mutual information for dataset Micro Organisms')

PLT.xlabel('Number of features')
PLT.ylabel('Precision (on scale [0, 1])')


PLT.show()