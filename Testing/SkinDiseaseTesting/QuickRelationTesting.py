from DataExtraction import SkinDataExtraction as SDE, GeneDataExtraction as GDE, ProcessExtraction as PE
from StatisticalAnalysisMethods import RelationTesting as RT
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

data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE78097', 'GSE14905', 'GSE32924', 'GSE27887', 'GSE36842']

suitable_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905']

testing = "Process"

# Extract the data
print("Extracting data...")
sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(suitable_names)
gene_set = GDE.extract_gene_data()

# Only use the significant values
print("Calculating significant values...")
sign_NL_values, sign_L_values, sign_gene_ids = SE.extract_significant_genes(sample_values, skin_type_values, gene_ids,
                                                                         target_groups=[1, 2], threshold=0.001,
                                                                         relation_groups='unequal variance', sorting=True)


# # Only use specific genes
# final_genes = []
# for gene in gene_set:
#     if gene_set[gene]['Representative Public ID'] in ["NM_002965", "NM_002964", "AW238654"]:
#         final_genes.append(gene)
#
# # Find processes with similar relation
# print("Searching for %s relations between genes..." % testing)
# sign_process = RT.testing_gene_relations(final_genes, gene_set, printing=2, relation_type=testing)

process, process_genes = PE.extract_processes(sign_gene_ids, gene_set)

testing_process = "negative regulation of intrinsic apoptotic signaling pathway in response to oxidative stress"

print("Process: %s" % testing_process)
for gene in process_genes[testing_process]:
    name = gene_set[gene]['Representative Public ID']
    description = gene_set[gene]["Gene Title"]
    print("Name: %s, Description: %s" % (name, description))
