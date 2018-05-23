from FeatureReductionMethods import FilterMethods as FM
from DataExtraction import DataSetExtraction as DSE

from sklearn import model_selection as SMS
from sklearn import linear_model as SLM
from sklearn import preprocessing as PP
from sklearn import metrics as SME

import numpy as np
import csv
import time

used_test = ["MI", "T-test"]
thresholds = [50, 100, 150]
used_data = ["MicroOrganisms", "Arcene", "Psoriasis", "RSCTC"]

test_name = []
data_name = []
threshold_name = []
order_name = []
T_val = []
T_test = []
T_feat = []
T_time = []
precision = []
recall = []
F1 = []

# T-tests
for name in used_data:
    for test in used_test:
        for threshold in thresholds:

            print("Currently at data %s, test %s and threshold %f" % (name, test, threshold))

            test_name.append('filter')
            data_name.append(name)
            order_name.append(test)

            # Extract data
            X, y, features = DSE.import_example_data(name)
            X = PP.normalize(X)

            # Split data in training and test data.
            X_train, X_test, y_train, y_test = SMS.train_test_split(X, y)

            start_time = time.clock()

            X_new, feat_new, alt_threshold = FM.filter_methods_classifier(X_train, y_train, features, ranking_method=test,
                                                           filter_method='Population',
                                                           threshold=threshold)

            threshold_name.append(alt_threshold)
            print("Alternative threshold is %f" % alt_threshold)

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

            print("Validation score is %f, test score is %f" % (T_val[-1], T_test[-1]))

with open('New_Filter_values.csv', 'w', newline='') as csv_file:
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




