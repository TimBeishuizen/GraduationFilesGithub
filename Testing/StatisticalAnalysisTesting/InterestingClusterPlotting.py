import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as SC
import sklearn.preprocessing as SP
import math

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
values = np.concatenate((np.array(sign_NL_values), np.array(sign_L_values)), axis=1)
scaler = SP.StandardScaler()
scaler.fit(values)
stand_values = scaler.transform(values)

clustering_type = "Aggl"
n_clust = 10

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

# Find all processes
print("Finding all significant %s aspects..." % testing)
process_genes = {}
processes = []

# Find cluster specific details (gene relations)
for i in range(n_clust):
    if i == 5 or i == 8:
        spec_gene_ids = np.extract(clust_labels == i, sign_gene_ids)
        for gene in spec_gene_ids:
            for process_relation in gene_set[gene]['%s Relations' % testing]:
                if len(process_relation) > 1:
                    if process_relation[1] not in processes:
                        processes.append(process_relation[1])
                        process_genes[process_relation[1]] = []

# Find all processes
print("Finding all genes for the %s aspects..." % testing)

for gene in sign_gene_ids:
    for process_relation in gene_set[gene]['%s Relations' % testing]:
        if len(process_relation) > 1:
            if process_relation[1] in processes:
                process_genes[process_relation[1]].append(gene)

# Averaging results
NL_avg = np.average(stand_values[:, :210], 1)
L_avg = np.average(stand_values[:, 210:], 1)

print("%i %s aspects are involved" % (len(processes), testing))

#Plotting the division of processes
print("Plotting the different %s aspects..." % testing)

frame_size = 4

for p in range(int(math.ceil(len(processes)/(frame_size * frame_size)))):
    f, plots = plt.subplots(frame_size, frame_size)
    for i in range(frame_size * frame_size):


        NL_process = []
        L_process = []

        NL_not_process = []
        L_not_process = []

        if (p * (frame_size * frame_size) + i) < len(processes):
            process = processes[p * (frame_size * frame_size) + i]

            for j in range(len(sign_gene_ids)):
                if sign_gene_ids[j] in process_genes[process]:
                    NL_process.append(NL_avg[j])
                    L_process.append(L_avg[j])
                else:
                    NL_not_process.append(NL_avg[j])
                    L_not_process.append(L_avg[j])

            plots[i % frame_size, math.floor(i / frame_size)].scatter(NL_not_process, L_not_process)
            plots[i % frame_size, math.floor(i / frame_size)].scatter(NL_process, L_process)
            plots[i % frame_size, math.floor(i / frame_size)].set_title("%s" % process)

    plt.show()
