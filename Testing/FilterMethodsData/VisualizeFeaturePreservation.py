import numpy as np
import csv
import matplotlib.pyplot as PLT

data_names = ['Psoriasis', 'RSCTC', 'Arcene', 'MO']
ranking_names = ['T', 'MI']
data_type = ['feat', 'test', 'val']

feat_complete = []
test_complete = []
val_complete = []
F1_complete = []
rec_complete = []
prec_complete = []
legend_complete = []

for used_data in data_names:
    for used_test in ranking_names:

        print("currently at %s and %s" % (used_data, used_test))

        feat = []
        test = []
        val = []
        prec = []
        rec = []
        F1 = []

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

        if used_data != 'MO':
            feat_complete.append(np.array(feat, dtype=int)[:, 1:])
            test_complete.append(np.array(test, dtype=float)[:, 1:])
            val_complete.append(np.array(val, dtype=float)[:, 1:])
        else:
            feat_complete.append(np.array(feat, dtype=int)[:, 1:-1])
            test_complete.append(np.array(test, dtype=float)[:, 1:-1])
            val_complete.append(np.array(val, dtype=float)[:, 1:-1])

        prec_complete.append(np.array(prec, dtype=float)[:, -11:])
        rec_complete.append(np.array(rec, dtype=float)[:, -11:])
        F1_complete.append(np.array(F1, dtype=float)[:, -11:])

feat_complete = np.asarray(feat_complete)
test_complete = np.asarray(test_complete)
val_complete = np.asarray(val_complete)
prec_complete = np.asarray(prec_complete)
rec_complete = np.asarray(rec_complete)
F1_complete = np.asarray(F1_complete)

print(prec_complete.shape)

feat_viz = []
test_viz = []
val_viz = []

line_arg = ['-g', '--g', '-b', '--b', '-y', '--y', '-r', '--r', '-c', '--c']

# for i in range(feat_complete.shape[1]):
#     feat_viz.append(np.mean(feat_complete[:, i, :], axis=0))
#     test_viz.append(np.mean(test_complete[:, i, :], axis=0))
#     val_viz.append(np.mean(val_complete[:, i, :], axis=0))
#     PLT.plot(feat_viz[i], val_viz[i], line_arg[2 * i])
# legend_final = ['Logistic Regression', 'Decision Tree', 'Nearest Neighbours', 'SVM', 'Naive Bayes']


for i in range(feat_complete.shape[0]):
    feat_mean = np.mean(feat_complete[i, :, :], axis=0)
    test_mean = np.mean(test_complete[i, :, :], axis=0)
    val_mean = np.mean(rec_complete[i, :, :], axis=0)

    # pre_val_mean = [0]
    # pre_val_mean.extend(val_mean[:-1])
    # pre_feat_mean = [0]
    # pre_feat_mean.extend(feat_mean[:-1])


    feat_viz.append(feat_mean)
    test_viz.append(test_mean)
    val_viz.append(val_mean)
    # val_viz.append((val_mean - pre_val_mean) / (feat_mean - pre_feat_mean))
    PLT.plot(feat_viz[i - 1], val_viz[i - 1], line_arg[i - 1])
legend_final = legend_complete


# feat_viz = np.mean(feat_complete[:, :, :], axis=tuple([0, 1])).tolist()
# test_viz = np.mean(test_complete[:, :, :], axis=tuple([0, 1])).tolist()
# val_viz = np.mean(val_complete[:, :, :], axis=tuple([0, 1])).tolist()
#
# print(test_viz)
#
# PLT.plot(feat_viz, val_viz, 'r')
# PLT.plot(feat_viz, test_viz, 'b')
# legend_final = ['Validation score', 'Test score']

# print(feat_viz)
PLT.legend(legend_final)
PLT.xlabel('Number of the feature added')
PLT.ylabel('Score quality addition ([0, 1])')
PLT.title('The quality addition of every feature after adding them to the dataset')

PLT.axis([0, 300, 0, 0.1])

PLT.show()


