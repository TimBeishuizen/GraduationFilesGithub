from metalearn import Metafeatures
from DataExtraction import DataSetExtraction as DSE
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

X, y, features = DSE.import_example_data('Hepatitis')

missing_values = ''

dfX = pd.DataFrame(X, columns=features)

dfX = dfX.replace(missing_values, np.NaN)

type = []
for header in list(dfX):
    X_f = dfX[header].values
    print(X_f)
    try:
        X_f.astype(float)
        type.append('NUMERICAL')
        dfX[header] = pd.to_numeric(dfX[header], errors='coerce')
    except:
        type.append('CATEGORICAL')
        dfX[header] = dfX[header].astype('category')

#dfX = dfX.apply(pd.Categorical, errors='ignore')

dfy = pd.Series(y, dtype='category')
dfy.name = "Output"

_metadata_path = "metafeatures_relevant.json"
with open(_metadata_path, 'r') as f:
    _metadata = json.load(f)
IDS = list(_metadata["metafeatures"].keys())

hot_encoded = pd.get_dummies(dfX)

testing_ids = ["ClassHistogram", "DistributionBoxplots", #"CategoricalCardinalityOutlierHistograms",
               "NumericCardinalityOutlierBoxplots", #"CategoricalAttributeEntropyOutlierHistograms",
               "NumericAttributeEntropyOutlierBoxplots", #"CategoricalJointEntropyOutlierHistograms",
               #"NumericJointEntropyOutlierHistograms", "CategoricalMutualInformationOutlierHistograms",
               ]#"NumericMutualInformationOutlierHistograms"]

mf = Metafeatures()
computations = mf.compute(dfX, dfy)#, metafeature_ids=testing_ids)

for key, value in computations.items():
    continue
    #print(key, value)