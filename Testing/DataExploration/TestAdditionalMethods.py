from metalearn import Metafeatures
from DataExtraction import DataSetExtraction as DSE
from DataExplorationMethods import information_theoretic_metafeatures_separate as ITM, common_operations as CO
import pandas as pd
import numpy as np
import os
import json

X, y, features = DSE.import_example_data('Hepatitis')

missing_values = ''

dfX = pd.DataFrame(X, columns=features)

dfX = dfX.replace(missing_values, np.NaN)

type = {}
for header in list(dfX):
    X_f = dfX[header].values
    cleaned_X_f = np.delete(X_f, np.argwhere(X_f == missing_values))
    try:
        cleaned_X_f.astype(float)
        type[header] = 'NUMERICAL'
        dfX[header] = pd.to_numeric(dfX[header], errors='coerce')
    except:
        type[header] = "CATEGORICAL"
        dfX[header] = dfX[header].astype('category')

#dfX = dfX.apply(pd.Categorical, errors='ignore')

dfy = pd.Series(y, dtype='category')
dfy.name = "Output"

mf = Metafeatures()

noNaN_cat_feat, = mf._get_categorical_features_with_no_missing_values(dfX, column_types=type)

entropies = ITM.get_separate_attribute_entropy(noNaN_cat_feat)
best_worst_entropies, locations = CO.return_most_important_attribute_entropies(entropies, return_end='Both', return_number=3)
print(best_worst_entropies)
print(locations)