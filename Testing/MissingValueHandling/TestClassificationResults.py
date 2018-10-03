import csv
import time

import numpy as np

from DataExtraction import DataSetExtraction as DSE
from MissingValueHandlingMethods import ListDeletionMethods as LDM, \
    SingleImputationMethods as SIM, \
    MultipleImputationMethods as MIM, PreprocessingMethods as HE

missing_value_methods = ['ACA']#['CCA', 'ACA', 'WCA', 'mean', 'hot deck', 'missing indicator mean', 'missing indicator zeros',
                        #'regression', 'kNN k=1', 'kNN k=3', 'kNN k=5',
                        #'MICE s=3, m=3', 'MICE s=3, m=5', 'MICE s=5, m=3', 'MICE s=5, m=5']
data_names = ['Hepatitis', 'Cirrhosis', 'Cervical']
# data_names = 'HeartAttack' needs regression

test_name = []
data_name = []
missing_value_method_name = []
T_val = []
T_test = []
T_time = []
precision = []
recall = []
F1 = []

for data in data_names:

    # Extract data
    X, y, features = DSE.import_example_data(data)

    for method in missing_value_methods:

        print("Currently at data %s combined with method %s" % (data, method))

        data_name.append(data)
        missing_value_method_name.append(method)

        new_y = y

        #X = LDM.aca(X, missing_values='', removal_fraction=0.15)

        start_time = time.clock()

        # Do missing value handling
        if method == 'CCA':
            test_name.append('LD')
            new_X, new_y = LDM.cca(X, y, missing_values='')

        elif method == 'ACA':
            test_name.append('LD')
            new_X, new_y = LDM.cca(LDM.aca(X, missing_values='', removal_fraction=0.001), y, missing_values='')

        elif method == 'WCA':
            test_name.append('LD')
            new_X = LDM.wca(X, missing_values='')
        elif method == 'mean':
            test_name.append('SI')
            new_X = SIM.mean_imputation(X, missing_values='')
        elif method == 'hot deck':
            test_name.append('SI')
            new_X = SIM.hot_deck_imputation(X, missing_values='')
        elif method == 'missing indicator mean':
            test_name.append('SI')
            new_X = SIM.mean_imputation(SIM.missing_indicator_imputation(X, missing_values=''), missing_values='')
        elif method == 'missing indicator zeros':
            test_name.append('SI')
            new_X = SIM.value_imputation(SIM.missing_indicator_imputation(X, missing_values=''), missing_values='', imputation_value=0)
        elif method == 'regression':
            test_name.append('SI')
            new_X = SIM.regression_imputation(X, missing_values='')
        elif 'kNN' in method:
            test_name.append('SI')
            new_X = SIM.kNN_imputation(X, missing_values='', k=int(method[6]))
        elif 'MICE' in method:
            test_name.append('MI')
            new_X = MIM.MICE(X, s=int(method[7]), m=int(method[12]), missing_values='')
        else:
            raise ValueError("%s not implemented in experiment" % method)

        elapsed = time.clock() - start_time
        T_time.append(elapsed)
        print("Time elapsed: %f" % elapsed)

        if not 'MICE' in method:

            new_X = HE.hot_encode_categorical_values(new_X)

            val, test, prec, rec, Fbeta = HE.compute_scores(new_X, new_y, nr_cross_val=10)
        else:

            tot_val = []
            tot_test = []
            tot_prec = []
            tot_rec = []
            tot_Fbeta = []

            for i in range(len(new_X)):
                temp_X = np.copy(new_X[i])

                temp_new_X = HE.hot_encode_categorical_values(temp_X)

                temp_val, temp_test, temp_prec, temp_rec, temp_Fbeta = HE.compute_scores(temp_new_X, new_y, nr_cross_val=10)

                tot_val.append(temp_val)
                tot_test.append(temp_test)
                tot_prec.append(temp_prec)
                tot_rec.append(temp_rec)
                tot_Fbeta.append(temp_Fbeta)

            val = np.mean(tot_val)
            test = np.mean(tot_test)
            prec = np.mean(tot_prec)
            rec = np.mean(tot_rec)
            Fbeta = np.mean(tot_Fbeta)

        # add scores
        T_val.append(val)
        T_test.append(test)
        precision.append(prec)
        recall.append(rec)
        F1.append(Fbeta)

        print("Validation score is %f, test score is %f" % (T_val[-1], T_test[-1]))

with open('MissingValueHandlingBlankTest.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(data_name)
    csv_writer.writerow(test_name)
    csv_writer.writerow(missing_value_method_name)
    csv_writer.writerow(T_val)
    csv_writer.writerow(T_test)
    csv_writer.writerow(T_time)
    csv_writer.writerow(precision)
    csv_writer.writerow(recall)
    csv_writer.writerow(F1)


