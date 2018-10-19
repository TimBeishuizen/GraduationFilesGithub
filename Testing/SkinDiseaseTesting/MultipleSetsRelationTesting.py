from DataExtraction import SkinDataExtraction as SDE
from DataExtraction import GeneDataExtraction as GDE
from StatisticalAnalysisMethods import SignificanceExtraction as SE
from StatisticalAnalysisMethods import RelationTesting as RT

import numpy as np

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

# Only use the significant values of non-lesional skin
print("Calculating significant values...")
sign_group_1, sign_group_2, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='unequal variance', sorting=True)

# # Only use the significant values of lesional skin
# print("Calculating significant values...")
# sign_group_1b, sign_group_2b, sign_gene_ids_b = \
#     SE.extract_significant_genes(sample_values, skin_type_values, gene_ids, target_groups=[0, 2], threshold=0.001,
#                                  relation_groups='unequal variance', sorting=True)
#
# # Remove the values present in lesional and non-lesional skin
# print("Remove the double values...")
# final_values, final_gene_ids = RT.remove_double_genes(np.concatenate((sign_group_1b, sign_group_2b), 1),
#                                                       sign_gene_ids_b, sign_gene_ids)

# Find processes with similar relation
print("Searching for process relations between genes...")
sign_process  = RT.testing_gene_relations(sign_gene_ids, gene_set, printing=30)