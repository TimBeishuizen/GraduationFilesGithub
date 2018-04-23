from FeatureReductionMethods import WrapperMethods as WM
from DataExtraction import DataSetExtraction as DSE
from sklearn import model_selection as SMS
from sklearn import linear_model as SLM
from sklearn import metrics as SME
import numpy as np
import csv
import time

used_data = ["Psoriasis"]
algorithms = ["floating"]
orders = ["Random", "MI"]
thresholds = [0.01, 0.001]

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
                print("Currently at test %i of 12, data %s, test %s, ordering %s, and threshold %f" % (index, name, test, order, threshold))
                index += 1

                data_name.append(name)
                threshold_name.append(threshold)
                test_name.append(test)
                order_name.append(order)

                # Extract data
                X, y, features = DSE.import_example_data(name)

                # Split data in training and test data.
                X_train, X_test, y_train, y_test = SMS.train_test_split(X, y)

                start_time = time.clock()
                if test == 'forward':
                    X_new, feat_new, _ = WM.forward_selection(X_train, y_train, features, ranking_method=order,
                                                           scoring_method='nb', improvement_threshold=threshold, scoring_cv=5)
                elif test == 'backward':
                    X_new, feat_new = WM.backward_selection(X_train, y_train, features, ranking_method=order,
                                                           scoring_method='nb', improvement_threshold=threshold, scoring_cv=5)
                elif type(test) == list:
                    X_new, feat_new = WM.sequential_search(X_train, y_train, features, sel_seq='FB', n_sel_seq=test,
                                                           scoring_method='nb', scoring_cv=5,
                                                           improvement_threshold=threshold,
                                                           ranking_method=order, continued_search=True)

                elif test == 'floating':
                    X_new, feat_new = WM.sequential_search(X_train, y_train, features, sel_seq='FB', n_sel_seq=[1300 - 1, 1300 - 1],
                                         scoring_method='nb', scoring_cv=5, improvement_threshold=threshold,
                                         ranking_method=order, continued_search=False)

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
                prec, rec, Fbeta, _ = SME.precision_recall_fscore_support(y_test, y_pred)
                precision.append(sum(prec)/len(prec))
                recall.append(sum(rec)/len(rec))
                F1.append(sum(Fbeta)/len(Fbeta))


                print("Validation score is %f, test score is %f" % (T_val[-1], T_test[-1]))

with open('Psoriasis_Wrapper_Float_values.csv', 'w', newline='') as csv_file:
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