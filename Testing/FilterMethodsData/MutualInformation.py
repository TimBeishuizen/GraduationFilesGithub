from DataExtraction import SkinDataExtraction as SDE, GeneDataExtraction as GDE
from sklearn import feature_selection as FE
import numpy as np

data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905', 'GSE32924', 'GSE27887', 'GSE36842']

suitable_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE14905']

# Extract multiple data sets
print("Extracting data...")
sample_values, skin_type_values, gene_ids = SDE.extract_multiple_data_sets(suitable_names)
gene_set = GDE.extract_gene_data()

val = FE.mutual_info_classif(sample_values, skin_type_values)

mi_ids = []
mi_val = []

for i in range(len(val)):
    if val[i] > 0.1:
        mi_ids.append(gene_ids[i])
        mi_val.append(val[i])

order = np.flip(np.argsort(mi_val), 0)
sort_val = [mi_val[i] for i in order]
sort_genes = [gene_ids[i] for i in order]

print("%i of the %i features had more than 0.1 mutual information" % (len(mi_val), len(val)))

for i in range(10):
    name = gene_set[sort_genes[i]]["Gene Title"]
    value = sort_val[i]
    print("%s has mutual information value %f" % (name, value))