import numpy as np
import sklearn.decomposition as SD
import sklearn.model_selection as SMS
import sklearn.preprocessing as SP
import sklearn.tree as ST

from DataExtraction import GeneDataExtraction as GDE
from DataExtraction import SkinDataExtraction as SDE
from StatisticalAnalysisMethods import MulticollinearityTesting as MT, SignificanceExtraction as SE

"""
Psoriasis              GSE13355          180         NN = Normal, PN = Non-Lesional, PP = Lesional
                       GSE30999          170         No normal patients
                       GSE34248          28          No normal patients
                       GSE41662          48          No normal patients
                       GSE78097          33          Different: Normal (0), Mild (1), Severe Psoriasis (2)
                       GSE14905          82                  
Atopic  dermatitis     GSE32924          33                  
                       GSE27887          35          Different: Pre NL (0), Post NL (1), Pre L (2), Post L (3)
                       GSE36842          39          Also tested difference between Acute (2) and Chronic (3) Dermatitis

"""

data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE78097', 'GSE14905', 'GSE32924', 'GSE27887', 'GSE36842']

suitable_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE14905']

# # Extract the data
# print("Extracting data...")
# sample_values, skin_type_values, gene_ids, sample_ids, series = SDE.extract_data(data_names[1])
# gene_set = GDE.extract_gene_data()

# Extract multiple data sets
print("Extracting data...")
sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(suitable_names)
gene_set = GDE.extract_gene_data()

# Only use the significant values of non-lesional skin
print("Calculating significant values...")
sign_group_1, sign_group_2, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='unequal variance', sorting=True)

# Scale the values
values = np.concatenate((np.array(sign_group_1), np.array(sign_group_2)), axis=1)
scaler = SP.StandardScaler()
scaler.fit(values)
stand_values = scaler.transform(values)

# Greedy cluster the genes
uncorr_values, uncorr_gene_ids, corr_values, corr_gene_ids = \
    MT.greedy_cluster_genes(stand_values[:2000,:], gene_ids, corr_threshold=0.7, info_print=False)

# Change gene IDs to names
uncorr_gene_names = []
for id in gene_ids:
    uncorr_gene_names.append(gene_set[id]['GB_ACC'])

# for i in range(len(uncorr_gene_ids)):
#     print("Gene ID cluster %i, name: %s" % (i, gene_set[uncorr_gene_ids[i]]['Gene Title']))

# Perform PCA
print("Performing PCA...")
pca = SD.PCA(n_components=0.95)
pca.fit(uncorr_values.transpose())
print(pca.explained_variance_ratio_)
print(pca.n_components_)

# Train test split the data
X_train, X_test, y_train, y_test = SMS.train_test_split(stand_values.T, np.concatenate(([1] * 210, [2]*213)))

# Find a random forest
print("Fitting in Decision Tree...")
rf = ST.DecisionTreeClassifier()
rf.fit(X_train, y_train)
print("Score of the Decision Tree: %s" % rf.score(X_test, y_test))
ST.export_graphviz(rf.tree_, out_file='tree_data.dot', feature_names=uncorr_gene_names, filled=True,
                          rounded=True)

