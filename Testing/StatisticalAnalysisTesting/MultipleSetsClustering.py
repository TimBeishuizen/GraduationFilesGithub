import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as SC
import sklearn.preprocessing as SP
from collections import Counter

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

data_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE78097','GSE14905','GSE32924','GSE27887', 'GSE36842']

suitable_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE14905']

# Extract the data
print("Extracting data...")
sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(suitable_names)
gene_set = GDE.extract_gene_data()

# Only use the significant values
print("Calculating significant values...")
sign_NL_values, sign_L_values, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='unequal variance', sorting=True)



# Scale the values
print("Scaling the significant values...")
print("%i Non-Lesional, %i Lesional skin samples" %(len(sign_NL_values[0]), len(sign_L_values[0])))
non_stand_values = np.concatenate((np.array(sign_NL_values), np.array(sign_L_values)), axis=1)
print(non_stand_values.shape)
print("values, Max: %f, Min: %f" % (np.max(non_stand_values), np.min(non_stand_values)))
scaler = SP.StandardScaler(with_mean=True)
stand_values = scaler.fit_transform(non_stand_values.T)
stand_values = stand_values.T

clustering_type = "Kmeans"
n_clust = 30

if clustering_type == "Kmeans":
    # Cluster using Kmeans
    print("Clustering using kMeans...")
    k_means = SC.KMeans(n_clust)
    k_means.fit(stand_values[:1000,:])
    clust_centers = k_means.cluster_centers_
    clust_labels = k_means.labels_
elif clustering_type == "DBSCAN":
    # Cluster using DBSCAN
    print("Clustering using DBSCAN...")
    dbscan = SC.DBSCAN()
    dbscan.fit(stand_values[:1000, :])
    clust_labels = dbscan.labels_
    n_clust = len(set(clust_labels))
    print(clust_labels)
elif clustering_type == "Aggl":
    # Cluster using Agglomerative clustering
    print("Clustering using Agglomerative clustering...")
    aggl_clust = SC.AgglomerativeClustering(n_clusters = n_clust)
    clust_labels = aggl_clust.fit_predict(stand_values[:, :])
    n_clust = len(set(clust_labels))
else:
    raise("Not implemented error")

# testing = "Process"
# testing = "Cellular"
testing = "Molecular"

# Find processes with similar relation
print("Searching for %s relations between genes..." % testing)
sign_process = []
relations = []
spec_gene_ids = []

# Find cluster specific details (gene relations)
for i in range(n_clust):
    print("Currently at cluster %i" % i)
    print("Cluster %i has %i different genes" % (i, sum(clust_labels == i)))
    spec_gene_ids.append(np.extract(clust_labels == i, sign_gene_ids))
    if i == -1:
        clust_sign_process = RT.testing_gene_relations(np.extract(clust_labels == i, sign_gene_ids), gene_set, printing=1, relation_type=testing)

        for j in range(len(spec_gene_ids)):
            print("Gene %i, name: %s, description: %s" %(j, gene_set[spec_gene_ids[i][j]]['Representative Public ID'], gene_set[spec_gene_ids[j]]['Gene Title']))
    else:
        clust_sign_process = RT.testing_gene_relations(np.extract(clust_labels == i, sign_gene_ids), gene_set,
                                                       printing=0.3, relation_type=testing)
    sign_process.append(clust_sign_process)

# Plotting a single one
f, solo_plot = plt.subplots()
for j in range(n_clust):
    NL_values = []
    L_values = []

    # Divide values in NL and L
    for k in range(stand_values.shape[1]):
        values = np.extract(clust_labels == j, stand_values[:, k])
        if k < 210:
            NL_values.append(values)
        elif k >= 210:
            L_values.append(values)

    # Average values
    NL_avg = np.average(np.array(NL_values), 0)
    L_avg = np.average(np.array(L_values), 0)
    solo_plot.scatter(NL_avg, L_avg)

    # Add labels for interesting values
    for i in range(NL_avg.shape[0]):
        gene_name = gene_set[spec_gene_ids[j][i]]["Representative Public ID"]
        if gene_name in ["U19557", "BC005224", "U19556"]:
            solo_plot.annotate('A', xy=(NL_avg[i], L_avg[i]), textcoords='offset points')
        elif gene_name in ["AF216693", "AW238654", "AJ001698", "M86849", "NM_005532", "NL_002638"]:
            solo_plot.annotate('B', xy=(NL_avg[i], L_avg[i]), textcoords='offset points')
        elif gene_name in ["NM_006945", "AF061812", "BG327863", "BF575466", "AI923984", "AB049591", "AB049591", "AB048288", "J00269", "NM_005987", "L42612", "AL569511", "NM_001878", "NM_003125", "NM_005130", "AJ243672", "NM_004942", "L10343", "NM_002638"]:
            solo_plot.annotate('C', xy=(NL_avg[i], L_avg[i]), textcoords='offset points')
    #     elif j == 5 or j == 8:
    #        solo_plot.annotate('D', xy=(NL_avg[i], L_avg[i]), textcoords='offset points')

# Add plot specifics
solo_plot.set_title('Clustering of expression difference between non-lesional and lesional skin')
# solo_plot.plot([-0.5, 0.5], [-0.5, 0.5])
clust_legend = [("Cluster %i " % i) for i in range(n_clust)]
# print(clust_legend.append("Average line"))
count_dict = Counter(clust_labels)
plt.legend(["Average line"] + [("Cluster %i with %i genes" % (i, count_dict[i])) for i in range(n_clust)])
plt.xlabel("Non-lesional values (avg expression)")
plt.ylabel("Lesional values (avg expression)")

plt.show()