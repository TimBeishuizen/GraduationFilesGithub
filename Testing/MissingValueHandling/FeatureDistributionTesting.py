from DataExtraction import DataSetExtraction as DSE
from MissingValueHandlingMethods import ListDeletionMethods as LDM, MultipleImputationMethods as MIM, \
    SingleImputationMethods as SIM
from StatisticalAnalysisMethods import FeatureDistributionMethods as FDM

import csv
import numpy as np

data_name = 'HeartAttack'

X, y, features = DSE.import_example_data(data_name)

missing_percentage = np.count_nonzero(X == '', axis=0).astype(float) / X.shape[0]

print(missing_percentage)
print(np.mean(missing_percentage))

CCA_X, CCA_y = LDM.cca(X, y, missing_values='')

#ACA_X, ACA_y = LDM.aca(X, y, missing_values='', important_features=[4, 5], removal_fraction=0.10)
WCA_X, WCA_y = LDM.wca(X, y, missing_values='')

MI_X = SIM.mean_imputation(X, missing_values='')
HDI_X = SIM.hot_deck_imputation(X, missing_values='')
NNI_X = SIM.kNN_imputation(X, missing_values='', k=3)
MRI_X = SIM.regression_imputation(X, missing_values='')

MICE_X = MIM.MICE(X, missing_values='', s=5, m=1)

All_names = ['1. CCA', '2. WCA', '3. Mean', '4. Hot deck', '5. kNN', '6. Regression', '7. MICE']

if True:
    CCA_p, CCA_var_p = FDM.compute_distribution_similarities(CCA_X, X, missing_values='')
    WCA_p, WCA_var_p = FDM.compute_distribution_similarities(WCA_X, X, missing_values='')

    MI_p, MI_var_p = FDM.compute_distribution_similarities(MI_X, X, missing_values='')
    HDI_p, HDI_var_p = FDM.compute_distribution_similarities(HDI_X, X, missing_values='')
    NNI_p, NNI_var_p = FDM.compute_distribution_similarities(NNI_X, X, missing_values='')
    MRI_p, MRI_var_p = FDM.compute_distribution_similarities(MRI_X, X, missing_values='')

    MICE_p, MICE_var_p = FDM.compute_distribution_similarities(MICE_X[0], X, missing_values='')

    total_p = np.asarray([CCA_p, WCA_p, MI_p, HDI_p, NNI_p, MRI_p, MICE_p])
    total_var_p = np.asarray([CCA_var_p, WCA_var_p, MI_var_p, HDI_var_p, NNI_var_p, MRI_var_p, MICE_var_p])
    # print(features)
    # print("MEAN")
    #print(total_p)
    # print("VARIANCE")
    # print(total_var_p)

    #FDM.plot_p_relation(p_val=total_p[2:], dist_names=All_names[2:], missing_percentage=missing_percentage)
    #FDM.plot_p_relation(p_val=total_var_p[2:], dist_names=All_names[2:], missing_percentage=missing_percentage)

    # Opening CSV file
    with open(data_name + '_p_val.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(features)
        csv_writer.writerow(missing_percentage)
        csv_writer.writerows(total_p)
        csv_writer.writerows(total_var_p)



if False:

    Regular_specs = FDM.compute_distribution_values(X, missing_values='')

    CCA_specs = FDM.compute_distribution_values(CCA_X, missing_values='')
    #ACA_specs = FDM.compute_distribution_values(ACA_X, missing_values='')
    WCA_specs = FDM.compute_distribution_values(WCA_X, missing_values='')

    MI_specs = FDM.compute_distribution_values(MI_X, missing_values='')
    HDI_specs = FDM.compute_distribution_values(HDI_X, missing_values='')
    NNI_specs = FDM.compute_distribution_values(NNI_X, missing_values='')
    MRI_specs = FDM.compute_distribution_values(MRI_X, missing_values='')

    MICE_specs = FDM.compute_distribution_values(MICE_X[0], missing_values='')

    All_X = [X, CCA_X, MI_X, MICE_X[0]]
    All_names = ['1. CCA', '2. WCA', '3. Mean', '4. Hot deck', '5. kNN', '6. Regression', '7. MICE']

    # All_X = [X, MI_X]
    # All_names = ['Regular', 'MI_X']

    #FDM.show_distribution_histograms(All_X, All_names, missing_values='')

    All_specs = [CCA_specs, WCA_specs, MI_specs, HDI_specs, NNI_specs, MRI_specs, MICE_specs]

    FDM.show_specs_distributions(All_specs, All_names, off_set=Regular_specs, missing_percentage=missing_percentage)
