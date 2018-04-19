import csv

from DataExtraction import DataSetExtraction as DSE
from FeatureReductionMethods import FilterMethodsTests as FMT

data_name = "MicroOrganisms"
rank_name = "MI"

X, y, features = DSE.import_example_data(data_name)

p_values = [0.01, 0.05, 0.1, 0.15, 0.2]
MI_values = [0.01, 0.05, 0.075, 0.1, 0.15, 0.2]
population = [1, 5, 10, 25, 50, 75, 100, 150, 250, 500, 1000]


val_lg, test_lg, feat_lg = FMT.test_filter_methods_classifier(X, y, features, filter_values=MI_values, ranking_method=rank_name, filter_method="Ranks", ML_algorithm='lr')

# val_lg, test_lg, feat_lg = FMT.test_filter_methods_classifier(X, y, features, filter_values=population, ranking_method=rank_name, filter_method="Population", ML_algorithm='lr')
#
# val_dt, test_dt, feat_dt = FMT.test_filter_methods_classifier(X, y, features, filter_values=population, ranking_method=rank_name, filter_method="Population", ML_algorithm='dt')
#
# val_nn, test_nn, feat_nn = FMT.test_filter_methods_classifier(X, y, features, filter_values=population, ranking_method=rank_name, filter_method="Population", ML_algorithm='nn')
#
# val_svm, test_svm, feat_svm = FMT.test_filter_methods_classifier(X, y, features, filter_values=population, ranking_method=rank_name, filter_method="Population", ML_algorithm='svm')
#
# val_nb, test_nb, feat_nb = FMT.test_filter_methods_classifier(X, y, features, filter_values=population, ranking_method=rank_name, filter_method="Population", ML_algorithm='nb')

# with open(data_name + '_' + rank_name + '_val_values.csv', 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(val_lg)
#     csv_writer.writerow(val_dt)
#     csv_writer.writerow(val_nn)
#     csv_writer.writerow(val_svm)
#     csv_writer.writerow(val_nb)

with open(data_name + '_' + rank_name + '_test_values.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(test_lg)
    csv_writer.writerow(test_dt)
    csv_writer.writerow(test_nn)
    csv_writer.writerow(test_svm)
    csv_writer.writerow(test_nb)

with open(data_name + '_' + rank_name + '_feat_values.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(feat_lg)
    csv_writer.writerow(feat_dt)
    csv_writer.writerow(feat_nn)
    csv_writer.writerow(feat_svm)
    csv_writer.writerow(feat_nb)
