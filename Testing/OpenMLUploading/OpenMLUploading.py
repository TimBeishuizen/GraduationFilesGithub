import os
import arff
import numpy as np
from DataExtraction import SkinDataExtraction as SDE

"""
Psoriasis              GSE13355          180         NN = Normal, PN = Non-Lesional, PP = Lesional
                       GSE30999          170         No normal patients
                       GSE34248          28          No normal patients
                       GSE41662          48          No normal patients
                       GSE78097          33          Different: Normal (0), Mild (2), Severe Psoriasis (3)
                       GSE14905          82                  
Atopic  dermatitis     GSE32924          33                  
                       GSE27887          35          Different: Pre NL (0), Post NL (1), Pre L (2), Post L (3)
                       GSE36842          39          Also tested difference between Acute (2) and Chronic (3) Dermatitis

"""

data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE78097', 'GSE14905', 'GSE32924', 'GSE27887', 'GSE36842']

for data_name in data_names:
    sample_values, skin_types, gene_ids, sample_ids, series = SDE.extract_data(data_name)

    dataset = series

    # Add attributes
    attributes = []
    for gene in gene_ids:
        attributes.append((gene, 'REAL'))

    print(sample_values.shape)
    break

    # Add skin-types to data
    print(data_name)
    print(skin_types)
    list_values = sample_values.tolist()

    # options for skin types, 0, 1, 2 and 3
    if data_name == 'GSE78097':
        skin_options = ['normal skin', '', 'mild lesional skin', 'severe lesional skin']
    elif data_name == 'GSE27887':
        skin_options = ['pre non-lesional', 'post non-lesional', 'pre lesional', 'post lesional']
    elif data_name == 'GSE36842':
        skin_options = ['normal skin', 'non-lesional skin', 'acute lesional skin', 'chronic lesional skin']
    else:
        skin_options = ['normal skin', 'non-lesional skin', 'lesional skin']

    attributes.append(('skin type', skin_options))

    for i in range(len(list_values)):
        list_values[i].append(skin_options[int(skin_types[i])])

    dataset['attributes'] = attributes
    dataset['relation'] = data_name
    dataset['data'] = list_values

    arff.dump(dataset, open(data_name + '.arff', 'w'))
