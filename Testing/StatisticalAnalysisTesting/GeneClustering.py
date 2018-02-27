import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as SC
import sklearn.preprocessing as SP

from DataExtraction import GeneDataExtraction as GDE
from DataExtraction import SkinDataExtraction as SDE
from StatisticalAnalysisMethods import RelationTesting as RT, SignificanceExtraction as SE

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

# Extract the data
print("Extracting data...")
sample_values, skin_type_values, gene_ids, sample_ids, series = SDE.extract_data(data_names[1])
gene_set = GDE.extract_gene_data()

# Only use the significant values
print("Calculating significant values...")
sign_NL_values, sign_L_values, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids, 0.001, sorting=True)

# Scale the values
print("Scaling the significant values...")
values = np.concatenate((np.array(sign_NL_values), np.array(sign_L_values)), axis=1)
scaler = SP.StandardScaler()
scaler.fit(values)
stand_values = scaler.transform(values)


# # Cluster using Kmeans
# n_clust = 10
# print("Clustering using kMeans...")
# k_means = SC.KMeans(n_clust)
# k_means.fit(stand_values[:1000,:])
# clust_centers = k_means.cluster_centers_
# clust_labels = k_means.labels_

# # Cluster using DBSCAN
# print("Clustering using DBSCAN...")
# dbscan = SC.DBSCAN()
# dbscan.fit(stand_values[:1000, :])
# clust_labels = dbscan.labels_
# n_clust = len(set(clust_labels))
# print(clust_labels)

# Cluster using Agglomerative clustering
print("Clustering using Agglomerative clustering...")
aggl_clust = SC.AgglomerativeClustering(n_clusters = 10)
clust_labels = aggl_clust.fit_predict(stand_values[:, :])
n_clust = len(set(clust_labels))

# Find processes with similar relation
print("Searching for process relations between genes...")
sign_process = []
relations = []

for i in range(n_clust):
    print("Currently at cluster %i" % i)
    clust_sign_process = RT.testing_gene_relations(np.extract(clust_labels == i, sign_gene_ids)[:2000], gene_set)
    sign_process.append(clust_sign_process)

# Plotting the sub-plots
print("Plotting the results...")
f, plots = plt.subplots(5, 17)
for i in range(int(stand_values.shape[1]/2)):
    x_plot = int(i/17)
    y_plot = i % 17
    for j in range(n_clust):
        NL_values = np.extract(clust_labels == j, stand_values[:, i])
        L_values = np.extract(clust_labels == j, stand_values[:, 85 + i])
        plots[x_plot, y_plot].scatter(NL_values, L_values)
        plots[x_plot, y_plot].set_title('%i' % i)

# Plotting a single one
f, solo_plot = plt.subplots()
for j in range(n_clust):
    NL_values = np.extract(clust_labels == j, stand_values[:, 0])
    L_values = np.extract(clust_labels == j, stand_values[:, 85 + 0])
    solo_plot.scatter(NL_values, L_values)
    solo_plot.set_title('Patient %i' % 0)
plt.legend([("Cluster %i " % i) for i in range(n_clust)])

plt.show()