from FeatureReduction import CovarianceComputation as CC
from DataExtraction import SkinDataExtraction as SDE, GeneDataExtraction as GDE
from sklearn import linear_model as LM
from sklearn import model_selection as MS
import numpy as np
import csv

data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905', 'GSE32924', 'GSE27887', 'GSE36842']


# Extract the data
print("Extracting data for data set %s" % data_names[1])
sample_values, skin_type_values, gene_ids, sample_ids, series = SDE.extract_data(data_names[1])
gene_set = GDE.extract_gene_data()

sample_values_2 = []
skin_type_values_2 = []

# Removed values
for i in range(len(skin_type_values)):
    if skin_type_values[i] in [1, 2]:
        sample_values_2.append(sample_values[i, :])
        skin_type_values_2.append(skin_type_values[i])

sample_values_2 = np.asarray(sample_values_2)

X_train, X_test, y_train, y_test = MS.train_test_split(sample_values_2, skin_type_values_2)

ridge = LM.RidgeClassifier()
ridge.fit(X_train, y_train)
print(ridge.score(X_test, y_test))
print(ridge.coef_)

print("Max: %f" % np.max(ridge.coef_))
print("Min: %f" % np.min(ridge.coef_))
print("Mean: %f" % np.mean(ridge.coef_))

sign = np.count_nonzero(np.abs(ridge.coef_) > 0.001)

print("Sign: %i" % sign)