from DataExtraction import DataSetExtraction as DSE

import numpy as np

from sklearn import feature_selection as FS

X, y, features = DSE.import_example_data('MicroOrganisms')

print(X.shape)
print(features.shape)
print(np.unique(y).shape)

