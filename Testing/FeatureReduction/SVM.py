from DataExtraction import SkinDataExtraction as SDE, GeneDataExtraction as GDE
from sklearn import svm as SVM
import numpy as np

data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905', 'GSE32924', 'GSE27887', 'GSE36842']

suitable_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE14905']

# Extract multiple data sets
print("Extracting data...")
sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(suitable_names)
gene_set = GDE.extract_gene_data()

sample_values_2 = []
skin_type_values_2 = []

# Removed values
for i in range(len(skin_type_values)):
    if skin_type_values[i] in [0, 1]:
        sample_values_2.append(sample_values[i, :])
        skin_type_values_2.append(skin_type_values[i])

sample_values_2 = np.asarray(sample_values_2)

print("Starting the fitting of the transform vector machine...")
svc = SVM.LinearSVC()

result = svc.fit(sample_values_2, skin_type_values_2)

print("Max: %f" % np.max(svc.coef_))
print("Min: %f" % np.min(svc.coef_))
print("Mean: %f" % np.mean(svc.coef_))

sign = np.count_nonzero(np.abs(svc.coef_) > 0.01)

print("Sign: %i" % sign)