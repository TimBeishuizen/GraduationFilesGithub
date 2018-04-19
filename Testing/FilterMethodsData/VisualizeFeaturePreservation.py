import numpy as np
import csv
import matplotlib.pyplot as PLT

data_names = ['Psoriasis', 'RSCTC', 'Arcene', 'MO']
ranking_names = ['T', 'MI']
data_type = ['feat', 'test', 'val']

feat_complete = []
test_complete = []
val_complete = []
legend_complete = []

for used_data in data_names:
    for used_test in ranking_names:

        print("currently at %s and %s" % (used_data, used_test))

        feat = []
        test = []
        val = []

        legend_complete.append(used_data + ' ' + used_test)

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

        if used_data != 'MO':
            feat_complete.append(np.array(feat, dtype=int)[:, 1:])
            test_complete.append(np.array(test, dtype=float)[:, 1:])
            val_complete.append(np.array(val, dtype=float)[:, 1:])
        else:
            feat_complete.append(np.array(feat, dtype=int)[:, 1:-1])
            test_complete.append(np.array(test, dtype=float)[:, 1:-1])
            val_complete.append(np.array(val, dtype=float)[:, 1:-1])

feat_complete = np.asarray(feat_complete)
test_complete = np.asarray(test_complete)
val_complete = np.asarray(val_complete)

feat_viz = []
test_viz = []
val_viz = []

line_arg = ['-g', '--g', '-b', '--b', '-y', '--y', '-r', '--r', '-c', '--c']

# for i in range(feat_complete.shape[1]):
#     feat_viz.append(np.mean(feat_complete[:, i, :], axis=0))
#     test_viz.append(np.mean(test_complete[:, i, :], axis=0))
#     val_viz.append(np.mean(val_complete[:, i, :], axis=0))
#     PLT.plot(feat_viz[i], val_viz[i], line_arg[2 * i])
# legend_final = ['Logistic Regression', 'Decision Tree', 'Nearest Neighbours', 'SVM', 'Naïve Bayes']


# for i in range(feat_complete.shape[0]):
#     feat_viz.append(np.mean(feat_complete[i, :, :], axis=0))
#     test_viz.append(np.mean(test_complete[i, :, :], axis=0))
#     val_viz.append(np.mean(val_complete[i, :, :], axis=0))
#     PLT.plot(feat_viz[i], val_viz[i], line_arg[i])
# legend_final = legend_complete


feat_viz = np.mean(feat_complete[:, :, :], axis=tuple([0, 1])).tolist()
test_viz = np.mean(test_complete[:, :, :], axis=tuple([0, 1])).tolist()
val_viz = np.mean(val_complete[:, :, :], axis=tuple([0, 1])).tolist()

print(test_viz)

PLT.plot(feat_viz, val_viz, 'r')
PLT.plot(feat_viz, test_viz, 'b')
legend_final = ['Validation score', 'Test score']

print(feat_viz)
PLT.legend(legend_final)
PLT.xlabel('Number of features preserved')
PLT.ylabel('Scores for validation')
PLT.title('The validation score by number of features preserved')

PLT.show()


