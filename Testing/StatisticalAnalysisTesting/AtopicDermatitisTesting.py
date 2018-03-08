import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as SC
import sklearn.preprocessing as SP
from collections import Counter

from DataExtraction import GeneDataExtraction as GDE, SkinDataExtraction as SDE, ProcessExtraction as PE
from StatisticalAnalysisMethods import RelationTesting as RT, SignificanceExtraction as SE
from VisualisationMethods import ExpressionDifferencePlotting as EDP

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

psoriasis_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905']
AD_names = ['GSE32924', 'GSE36842']
Specific_AD_name = 'GSE27887'

testing = "Process"

# Extract the data
print("Extracting data...")
sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(AD_names)
gene_set = GDE.extract_gene_data()

sample_values_2, skin_type_values_2, gene_ids_2, sample_IDs, series = SDE.extract_data('GSE27887')
group_values_2 = SE.extract_target_groups(sample_values_2, skin_type_values_2, [0, 2])

sample_values = np.append(sample_values, np.asarray(group_values_2[0]), axis=0)
skin_type_values = np.append(skin_type_values, [1]*len(group_values_2[0]))

sample_values = np.append(sample_values, np.asarray(group_values_2[1]), axis=0)
skin_type_values = np.append(skin_type_values, [2]*len(group_values_2[1]))

# Only use the significant values of non-lesional skin
print("Calculating significant values...")
sign_values, sign_skin_types, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='unequal variance', sorting=True)

# Scale the values
print("Scaling the significant values...")
values = np.array(sign_values)
scaler = SP.StandardScaler()
stand_values = scaler.fit_transform(values.T).T

# Extract the data
print("Extracting data...")
ps_sample_values, ps_skin_type_values, ps_gene_ids = SDE.extract_multiple_data_sets(psoriasis_names)

# Only use the significant values of non-lesional skin
print("Calculating significant values...")
ps_sign_values, ps_sign_skin_types, ps_sign_gene_ids = SE.extract_significant_genes(ps_sample_values, ps_skin_type_values, ps_gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='unequal variance', sorting=True)

joint_genes = []

for gene_id in sign_gene_ids:
    if gene_id in ps_sign_gene_ids:
        joint_genes.append(gene_id)

print("There are %i joint significant genes" % len(joint_genes))
print(joint_genes)

# Find processes with similar relation
print("Searching for process relations between genes...")
sign_process = RT.testing_gene_relations(joint_genes, gene_set, printing=2)

# Find all processes
print("Finding all %s relations..." % testing)
processes, process_genes = PE.extract_processes(joint_genes, gene_set, relation_type=testing)

# Averaging results
NL_avg = np.average(stand_values[:, :28], 1)
L_avg = np.average(stand_values[:, 28:], 1)

# # Find significance between processes
# print("Remove insignificant %s aspects..." % testing)
# processes, sign_values = RT.remove_insignificant_processes(processes, process_genes, sign_gene_ids, L_avg, NL_avg,
#                                                            ordering=True, sign_threshold=0.5)

# # Plotting the division of processes
# print("Plotting the different %s aspects..." % testing)
# EDP.plot_multiple_processes(processes, process_genes, sign_gene_ids, L_avg, NL_avg, sign_values=sign_values, frame_size=4)