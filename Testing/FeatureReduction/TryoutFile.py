import matplotlib.pyplot as PLT

import numpy as np
import csv

from sklearn import model_selection as MS
from sklearn import naive_bayes as NB

from DataExtraction import DataSetExtraction as DSE
from FeatureReductionMethods import WrapperMethods as WM

data_name = 'Psoriasis'

X, y, features = DSE.import_example_data(data_name)

print("This data set has %i classes" % np.unique(y).shape[0])

X_train, X_test, y_train, y_test = MS.train_test_split(X, y, train_size=0.8)

print("Starting score of complete feature set: %f" % WM.compute_score(X, y, scoring_method='nb'))

# new_X, new_features = WM.forward_selection(X, y, features, improvement_threshold=0.001, scoring_method="dt", number_threshold=100, ranking_method="Mutual information")

# new_X, new_features = WM.backward_selection(X, y, features, improvement_threshold=0.01, scoring_method="dt", number_threshold=100) #, ranking_method="Mutual information")

new_X, new_features = WM.sequential_search(X_train, y_train, features, sel_seq='FB', n_sel_seq=[10, 5], scoring_method='nb',
                                           improvement_threshold=0.01, ranking_method='Mutual information', scoring_cv=8, max_iter=100, continued_search=True)


for i in reversed(range(features.shape[0])):
    if features[i] not in new_features.tolist():
        features = np.delete(features, i)
        X_test = np.delete(X_test, i, axis=1)

print("Old shape: %i, new shape %i" % (X.shape[1], new_X.shape[1]))
print("X-test:")
print(X_test.shape)

nb = NB.GaussianNB()
nb.fit(new_X, y_train)
print(nb.score(X_test, y_test))

print(np.corrcoef(new_X.T))

with open('candidate_X.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(new_features)
    csv_writer.writerows(new_X)