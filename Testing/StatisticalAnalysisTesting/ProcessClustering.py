import numpy as np
import sklearn.preprocessing as SP

from VisualisationMethods import ExpressionDifferencePlotting as EDP
from DataExtraction import GeneDataExtraction as GDE, SkinDataExtraction as SDE, ProcessExtraction as PE
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
sign_values, sign_skin_types, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='unequal variance', sorting=True)

# Scale the values
print("Scaling the significant values...")
values = np.array(sign_values)
scaler = SP.StandardScaler(with_mean=False)
stand_values = scaler.fit_transform(values.T).T

# testing = "Process"
# testing = "Cellular"
testing = "Molecular"

# Find all processes
print("Finding all %s relations..." % testing)
processes, process_genes = PE.extract_processes(sign_gene_ids, gene_set, relation_type=testing)

# Averaging results
NL_avg = np.average(stand_values[:, :210], 1)
L_avg = np.average(stand_values[:, 210:], 1)

# Find significance between processes
print("Remove insignificant %s aspects..." % testing)
processes, sign_values = RT.remove_insignificant_processes(processes, process_genes, sign_gene_ids, L_avg, NL_avg,
                                                           ordering=True, sign_threshold=0.01)

# Plotting the division of processes
print("Plotting the different %s aspects..." % testing)
EDP.plot_multiple_processes(processes, process_genes, sign_gene_ids, L_avg, NL_avg, frame_size=4)