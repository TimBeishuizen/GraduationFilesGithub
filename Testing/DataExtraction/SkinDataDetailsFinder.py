from DataExtraction import SkinDataExtraction as DE
from collections import Counter

data_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE78097','GSE14905','GSE32924','GSE27887', 'GSE36842']

samples = []
features = []



for i in range(len(data_names)):
    sample_values, skin_types, gene_ids, sample_ids = DE.extract_data(data_names[i])
    samples.append(sample_values.shape[0])
    features.append(sample_values.shape[1])
    print(data_names[i])
    print(Counter(skin_types))

print(samples)
print(features)