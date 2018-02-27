from DataExtraction import SkinDataExtraction as SDE
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

samples = []
skin_types = []

print("Extracting data...")
for i in range(len(data_names)):
    print("Currently at data set %s" % data_names[i])
    sample_values, skin_type_values, gene_ids, sample_ids = SDE.extract_data(data_names[i])
    samples.append(sample_values)
    skin_types.append(skin_type_values)
    print("Sample size: %i" % sample_values.shape[0])
    print("Number of skin types: %i" % np.unique(skin_type_values).shape[0])