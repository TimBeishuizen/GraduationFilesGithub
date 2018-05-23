import numpy as np
import time

from sklearn import model_selection as SMS

from FeatureReductionMethods import WrapperMethods as WM
from DataExtraction import DataSetExtraction as DSE

# Extract data
X, y, features = DSE.import_example_data('MicroOrganisms')

# Split data in training and test data.
X_train, X_test, y_train, y_test = SMS.train_test_split(X, y)

start_time = time.clock()

X_new, feat_new= WM.simulated_annealing(X_train, y_train, features, scoring_method='nb', T0=1, T1=0.01, m=10000, v=0.98,
                                        cv=5, penalty=0.01, subset_size=100, p_selected=0.5)

