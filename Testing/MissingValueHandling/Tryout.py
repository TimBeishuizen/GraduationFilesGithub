from DataExtraction import DataSetExtraction as DSE
from MissingValueHandlingMethods import ListDeletionMethods as LDM, SingleImputationMethods as SIM, \
    MultipleImputationMethods as MIM

import numpy as np

X, y, features = DSE.import_example_data('Cirrhosis')

print(X.shape)
print(y.shape)

CCA_X, CCA_y = LDM.cca(X, y, missing_values='')

ACA_X, ACA_y = LDM.aca(X, y, missing_values='', important_features=[4, 5], removal_fraction=0.10)

WCA_X, WCA_y = LDM.wca(X, y, missing_values='')

MI_X, MI_y = SIM.mean_imputation(X, y, missing_values='', imputation_type='mean')

HDI_X, HDI_y = SIM.hot_deck_imputation(X, y, missing_values='')

MII_X, MII_y = SIM.missing_indicator_imputation(X, y, missing_values='')
MII_X2, MII_y2 = SIM.value_imputation(MII_X, MII_y, missing_values='', imputation_value=0)

NNI_X, NNI_y = SIM.kNN_imputation(X, y, missing_values='', k=3)

MRI_X, MRI_y = SIM.regression_imputation(X, y, missing_values='')

MICE_X, MICE_y = MIM.MICE(X, y, missing_values='', s=1, m=1)

print(np.argwhere(X == ''))
print(np.argwhere(MICE_X == ''))