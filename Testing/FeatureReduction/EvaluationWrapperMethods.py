from FeatureReductionMethods import WrapperMethods as WM
from DataExtraction import DataSetExtraction as DSE
from sklearn import model_selection as SMS
from sklearn import linear_model as SLM
import numpy as np
import csv

used_data = ["MicroOrganisms"]
algorithms = ["forward", "backward"]
orders = ["Random", "MI"]
thresholds = [0.01, 0.001]

test_name = []
data_name = []
order_name = []
threshold_name = []
T_val = []
T_test = []
T_feat = []

for name in used_data:
    for test in algorithms:
        for order in orders:
            for threshold in thresholds:
                print("Currently at data %s, test %s, ordering %s, and threshold %f" % (name, test, order, threshold))

                data_name.append(name)
                threshold_name.append(threshold)
                test_name.append(test)
                order_name.append(order)

                # Extract data
                X, y, features = DSE.import_example_data(name)

                # Split data in training and test data.
                X_train, X_test, y_train, y_test = SMS.train_test_split(X, y)

                if test == 'forward':
                    X_new, feat_new, _ = WM.forward_selection(X_train, y_train, features, ranking_method=order,
                                                           scoring_method='nb', improvement_threshold=threshold, scoring_cv=8)
                elif test == 'backward':
                    X_new, feat_new = WM.backward_selection(X_train, y_train, features, ranking_method=order,
                                                           scoring_method='nb', improvement_threshold=threshold, scoring_cv=8)

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

                print("Validation score is %f, test score is %f" % (T_val[-1], T_test[-1]))

with open('Wrapper_Forward_Backward_values.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(data_name)
    csv_writer.writerow(test_name)
    csv_writer.writerow(threshold_name)
    csv_writer.writerow(order_name)
    csv_writer.writerow(T_val)
    csv_writer.writerow(T_test)
    csv_writer.writerow(T_feat)