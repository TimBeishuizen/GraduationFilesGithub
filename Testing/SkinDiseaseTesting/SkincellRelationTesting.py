from DataExtraction import SkinDataExtraction as SDE, GeneDataExtraction as GDE, ProcessExtraction as PE
from StatisticalAnalysisMethods import RelationTesting as RT
from StatisticalAnalysisMethods import SignificanceExtraction as SE
from VisualisationMethods import ExpressionDifferencePlotting as EDP
import numpy as np
import sklearn.preprocessing as SP

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

suitable_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905']

keratinocytes = ['NM_006945',  'BF575466', 'AI923984', 'AB049591', 'AF216693', 'AB048288', 'J00269',
                 'NM_005987', 'L42612', 'AL569511', 'NM_003125', 'NM_005130']
granulocytes = ['BG327863', 'AB049591', 'AB048288', 'NM_002965']
neutrophils = ['AW238654', 'NM_002964', 'NM_000045', 'NM_004942']
monocytes = ['AW238654', 'NM_002964', 'NM_002965']
epidermis = ['AW238654', 'NM_002964', 'AL569511']

testing = "Process"

current_testing = monocytes

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

NL_avg = np.average(stand_values[:, :210], axis=1)
L_avg = np.average(stand_values[:, 210:], axis=1)

# Only use specific genes
print("Using specific genes...")
final_genes = []
for gene in gene_set:
    if gene_set[gene]['Representative Public ID'] in current_testing:
        final_genes.append(gene)

# # Find processes with similar relation
print("Searching for %s relations between genes..." % testing)
sign_process = RT.testing_gene_relations(final_genes, gene_set, printing=2, relation_type=testing)

EDP.plot_gene_set(sign_gene_ids, final_genes, "monocytes", L_avg, NL_avg)

