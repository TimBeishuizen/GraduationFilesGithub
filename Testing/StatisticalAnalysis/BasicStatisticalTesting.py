from DataExtraction import SkinDataExtraction as SDE
from StatisticalAnalysis import SignificanceExtraction as SE
import numpy as np
import sklearn.decomposition as SD
import sklearn.preprocessing as SP
import time
import sklearn.ensemble as SEN

import math

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
print("Extract data...")
sample_values, skin_type_values, gene_ids, sample_ids = SDE.extract_data(data_names[1])

# Only use the significant values
print("Calculate significant values...")
sign_NL_values, sign_L_values, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids, 0.001, sorting=True)

# Scale the values
values = np.concatenate((np.array(sign_NL_values), np.array(sign_L_values)), axis=1)
scaler = SP.StandardScaler()
scaler.fit(values)
stand_values = scaler.transform(values)

print(stand_values.shape)

uncorr_values = []
corr_values = []
uncorr_gene_ids = []
corr_gene_ids = []
uncorrelated = 0

start_time = time.time()

# Check for multicollinearity
print("Checking for multicollinearity...")
for i in range(stand_values.shape[0]):
    if i%500==0:
        print("Currently at gene %i, %i uncorrelated genes found in %i seconds" % (i, uncorrelated, (time.time() - start_time)))
    for j in range(len(uncorr_values)):
        pearson_correlation = np.corrcoef(stand_values[i, :], uncorr_values[j])[0, 1]
        if abs(pearson_correlation) > 0.7:
            corr_values[j].append(stand_values[i,:])
            corr_gene_ids[j].append(gene_ids[i])
            break
    else:
        uncorr_values.append(stand_values[i, :])
        corr_values.append([stand_values[i, :]])
        uncorr_gene_ids.append(gene_ids[i])
        corr_gene_ids.append([gene_ids[i]])
        uncorrelated += 1

print("%i clusters are found:" % len(corr_gene_ids))
for i in range(len(corr_gene_ids)):
    correlation = np.min(abs(np.corrcoef(corr_values[i])))
    if len(corr_gene_ids[i]) > 10:
        print("Cluster: %i, number of genes: %i, minimum correlation: %f" % (i+1, len(corr_gene_ids[i]), correlation))

uncorr_values = np.asarray(uncorr_values)

# Perform PCA
print("Performing PCA...")
pca = SD.PCA(n_components=0.95)
pca.fit(uncorr_values.transpose())
print(pca.explained_variance_ratio_)
print(pca.n_components_)
