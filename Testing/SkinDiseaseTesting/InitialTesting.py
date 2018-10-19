from DataExtraction import SkinDataExtraction as SDE
import numpy as np
from DataExtraction import GeneDataExtraction as GDE
from StatisticalAnalysisMethods import SignificanceExtraction as SE

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
data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905', 'GSE32924', 'GSE27887', 'GSE36842']

suitable_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE14905']

for i in range(8):


    # Extract the data
    print("Extracting data for data set %s" % data_names[i])
    sample_values, skin_type_values, gene_ids, sample_ids, series = SDE.extract_data(data_names[i])
    gene_set = GDE.extract_gene_data()

    for gene in gene_ids:
        if gene_set[gene]["GB_ACC"] == 'AI286239':
            print('TITLE')
            print(gene_set[gene]["Process Relations"])
            # print(gene_set[gene]["Target Description"])
            print(gene_set[gene]["Cellular Relations"])
    break

# # Extract multiple data sets
# print("Extracting data...")
# sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(suitable_names)
# gene_set = GDE.extract_gene_data()

    # Only use the significant values of non-lesional skin
    print("Calculating significant values...")
    sign_group_1, sign_group_2, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='paired', sorting=True)

    print(len(sign_gene_ids))

if False:
    # Extract multiple data sets
    print("Extracting data...")
    sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(suitable_names)
    gene_set = GDE.extract_gene_data()

    sign_group_1, sign_group_2, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                             target_groups=[1, 2], threshold=0.001,
                                                                             relation_groups='unequal variance',
                                                                             sorting=True)

    print(len(sign_gene_ids))