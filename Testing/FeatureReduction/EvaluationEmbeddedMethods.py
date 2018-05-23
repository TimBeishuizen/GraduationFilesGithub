from FeatureReductionMethods import EmbeddedMethods as EM
from DataExtraction import DataSetExtraction as DSE
from sklearn import model_selection as SMS
from sklearn import linear_model as SLM
from sklearn import metrics as SME
from sklearn import preprocessing as PP
import numpy as np
import csv
import time

used_data = ["MicroOrganisms", "Arcene", "RSCTC", "Psoriasis", ]
algorithms = ["forward"] # , "backward"]
orders = ["svm", 'rf']
thresholds = [50, 100, 150]

test_name = []
data_name = []
order_name = []
threshold_name = []
T_val = []
T_test = []
T_feat = []
T_time = []
precision = []
recall = []
F1 = []

index = 1

for name in used_data:
    for test in algorithms:
        for order in orders:
            for threshold in thresholds:
                print("Currently at test %i of %i, data %s, test %s, ordering %s, and threshold %f" % (index, len(used_data) * len(algorithms) * len(orders) * len(thresholds),name, test, order, threshold))
                index += 1

                # Extract data
                X, y, features = DSE.import_example_data(name)

                X = PP.normalize(X)

                data_name.append(name)
                test_name.append(test)
                order_name.append(order)

                # Split data in training and test data.
                X_train, X_test, y_train, y_test = SMS.train_test_split(X, y)

                start_time = time.clock()
                if test == 'forward':

                    X_new, feat_new, alt_threshold = EM.embedded_forward_selection(X_train, y_train, features, ML_alg=order,
                                                                       filter_method="Population", threshold=threshold)
                elif test == 'backward':
                    X_new, feat_new, alt_threshold = EM.embedded_backward_elimination(X_train, y_train, features, ML_alg=order,
                                                                    filter_method="Population", threshold=threshold)

                threshold_name.append(alt_threshold)
                print("The alternative threshold is %f" % alt_threshold)

                elapsed = time.clock() - start_time
                T_time.append(elapsed)
                print("Time elapsed: %f" % elapsed)
                print("number of features is %i" % len(feat_new))
                T_feat.append(len(feat_new))

                # Creating test set
                X_test_new = []
                for feat in feat_new:
                    X_test_new.append(X_test[:, list(features).index(feat)])

                X_test_new = np.asarray(X_test_new).T

                # Cross validation data
                cross_val = SMS.LeaveOneOut()
                # cross_val = SMS.KFold(n_splits=10)
                val_score = []

                print("Starting cross validation")
                # Compute validation score
                for train_index, test_index in cross_val.split((X_new)):
                    ml = SLM.LogisticRegression()
                    ml.fit(X_new[train_index], y_train[train_index])
                    val_score.append(ml.score(X_new[test_index], y_train[test_index]))

                # Compute the mean of the validation score
                T_val.append(np.mean(np.asarray(val_score)))

                # Compute test score
                ml = SLM.LogisticRegression()
                ml.fit(X_new, y_train)
                T_test.append(ml.score(X_test_new, y_test))
                y_pred = ml.predict(X_test_new)
                prec, rec, Fbeta, _ = SME.precision_recall_fscore_support(y_test, y_pred, average='weighted')

                precision.append(prec)
                recall.append(rec)
                F1.append(Fbeta)

                print("Validation score is %f,  test score is %f, precision is %f, recall is %f, F1 is %f" % (T_val[-1], T_test[-1], precision[-1], recall[-1], F1[-1]))

with open('New_Embedded_values.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(data_name)
    csv_writer.writerow(test_name)
    csv_writer.writerow(threshold_name)
    csv_writer.writerow(order_name)
    csv_writer.writerow(T_val)
    csv_writer.writerow(T_test)
    csv_writer.writerow(T_feat)
    csv_writer.writerow(T_time)
    csv_writer.writerow(precision)
    csv_writer.writerow(recall)
    csv_writer.writerow(F1)