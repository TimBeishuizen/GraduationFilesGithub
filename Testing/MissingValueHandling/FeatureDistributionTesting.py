from DataExtraction import DataSetExtraction as DSE
from MissingValueHandlingMethods import ListDeletionMethods as LDM, MultipleImputationMethods as MIM, \
    SingleImputationMethods as SIM
from StatisticalAnalysisMethods import FeatureDistributionMethods as FDM

import csv
import numpy as np

data_names = ['HeartAttack', 'Hepatitis', 'Cirrhosis', 'Cervical']

alg_types = ['cca', 'wca', 'mean', 'hot_deck', 'regression', '1NN', '3NN', '5NN', 'MICE s=1', 'MICE s=3', 'MICE s=5']

data = []
feat = []
missing = []
mean_p = np.zeros((11, 0))
var_p = np.zeros((11, 0))

for data_name in data_names:

    X, y, features = DSE.import_example_data(data_name)

    missing_percentage = np.count_nonzero(X == '', axis=0).astype(float) / X.shape[0]

    ACA = False
    if ACA:
        X = LDM.aca(X, missing_values='', removal_fraction=0.15)
        for i in reversed(range(missing_percentage.shape[0])):
            if missing_percentage[i] > 0.15:
                print("Deleting")
                features = np.delete(features, i, 0)
                missing_percentage = np.delete(missing_percentage, i, 0)

    print("Gathering new datasets...")

    CCA_X, CCA_y = LDM.cca(X, y, missing_values='')
    WCA_X = LDM.wca(X, missing_values='')

    MI_X = SIM.mean_imputation(X, missing_values='')
    HDI_X = SIM.hot_deck_imputation(X, missing_values='')
    print("Starting with nearest neighbours...")
    NN1I_X = SIM.kNN_imputation(X, missing_values='', k=1)
    NN3I_X = SIM.kNN_imputation(X, missing_values='', k=3)
    NN5I_X = SIM.kNN_imputation(X, missing_values='', k=5)
    print("Starting with regression...")
    MRI_X = SIM.regression_imputation(X, missing_values='')

    print("Gathering p-values...")

    CCA_p, CCA_var_p = FDM.compute_distribution_similarities(CCA_X, X, missing_values='')
    WCA_p, WCA_var_p = FDM.compute_distribution_similarities(WCA_X, X, missing_values='')

    MI_p, MI_var_p = FDM.compute_distribution_similarities(MI_X, X, missing_values='')
    HDI_p, HDI_var_p = FDM.compute_distribution_similarities(HDI_X, X, missing_values='')
    NN1I_p, NN1I_var_p = FDM.compute_distribution_similarities(NN1I_X, X, missing_values='')
    NN3I_p, NN3I_var_p = FDM.compute_distribution_similarities(NN3I_X, X, missing_values='')
    NN5I_p, NN5I_var_p = FDM.compute_distribution_similarities(NN5I_X, X, missing_values='')
    MRI_p, MRI_var_p = FDM.compute_distribution_similarities(MRI_X, X, missing_values='')

    MICE1_p_temp = []
    MICE3_p_temp = []
    MICE5_p_temp = []
    MICE1_var_p_temp = []
    MICE3_var_p_temp = []
    MICE5_var_p_temp = []

    print("Start with MICE data gathering...")

    for i in range(10):

        MICE1_X_temp = MIM.MICE(X, missing_values='', s=1, m=1)
        MICE3_X_temp = MIM.MICE(X, missing_values='', s=3, m=1)
        MICE5_X_temp = MIM.MICE(X, missing_values='', s=5, m=1)

        MICE1_p_cur, MICE1_var_p_cur = FDM.compute_distribution_similarities(MICE1_X_temp[0], X, missing_values='')
        MICE3_p_cur, MICE3_var_p_cur = FDM.compute_distribution_similarities(MICE3_X_temp[0], X, missing_values='')
        MICE5_p_cur, MICE5_var_p_cur = FDM.compute_distribution_similarities(MICE5_X_temp[0], X, missing_values='')

        MICE1_p_temp.append(MICE1_p_cur), MICE3_p_temp.append(MICE3_p_cur), MICE5_p_temp.append(MICE5_p_cur)
        MICE1_var_p_temp.append(MICE1_var_p_cur), MICE3_var_p_temp.append(MICE3_var_p_cur), MICE5_var_p_temp.append(MICE5_var_p_cur)

    MICE1_p = np.mean(np.asarray(MICE1_p_temp), axis=0)
    MICE3_p = np.mean(np.asarray(MICE3_p_temp), axis=0)
    MICE5_p = np.mean(np.asarray(MICE5_p_temp), axis=0)
    MICE1_var_p = np.mean(np.asarray(MICE1_var_p_temp), axis=0)
    MICE3_var_p = np.mean(np.asarray(MICE3_var_p_temp), axis=0)
    MICE5_var_p = np.mean(np.asarray(MICE5_var_p_temp), axis=0)

    total_p = np.asarray([CCA_p, WCA_p, MI_p, HDI_p, MRI_p, NN1I_p, NN3I_p, NN5I_p, MICE1_p, MICE3_p, MICE5_p])
    total_var_p = np.asarray([CCA_var_p, WCA_var_p, MI_var_p, HDI_var_p, MRI_var_p, NN1I_var_p, NN3I_var_p, NN5I_var_p,
                              MICE1_var_p, MICE3_var_p, MICE5_var_p])


    #FDM.plot_p_relation(p_val=total_p[2:], dist_names=All_names[2:], missing_percentage=missing_percentage)
    #FDM.plot_p_relation(p_val=total_var_p[2:], dist_names=All_names[2:], missing_percentage=missing_percentage)

    data.extend([data_name] * len(alg_types) * len(CCA_p))
    feat.extend(features)
    missing.extend(missing_percentage)
    mean_p = np.append(mean_p, total_p, axis=1)
    var_p = np.append(var_p, total_var_p, axis=1)


# Opening CSV file
with open('combined_p_val.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(data)
    csv_writer.writerow(missing)
    csv_writer.writerow(feat)
    csv_writer.writerows(mean_p.tolist())
    csv_writer.writerows(var_p.tolist())


#
# if False:
#
#     Regular_specs = FDM.compute_distribution_values(X, missing_values='')
#
#     CCA_specs = FDM.compute_distribution_values(CCA_X, missing_values='')
#     #ACA_specs = FDM.compute_distribution_values(ACA_X, missing_values='')
#     WCA_specs = FDM.compute_distribution_values(WCA_X, missing_values='')
#
#     MI_specs = FDM.compute_distribution_values(MI_X, missing_values='')
#     HDI_specs = FDM.compute_distribution_values(HDI_X, missing_values='')
#     NNI_specs = FDM.compute_distribution_values(NNI_X, missing_values='')
#     MRI_specs = FDM.compute_distribution_values(MRI_X, missing_values='')
#
#     MICE_specs = FDM.compute_distribution_values(MICE_X[0], missing_values='')
#
#     All_X = [X, CCA_X, MI_X, MICE_X[0]]
#     All_names = ['1. CCA', '2. WCA', '3. Mean', '4. Hot deck', '5. kNN', '6. Regression', '7. MICE']
#
#     # All_X = [X, MI_X]
#     # All_names = ['Regular', 'MI_X']
#
#     #FDM.show_distribution_histograms(All_X, All_names, missing_values='')
#
#     All_specs = [CCA_specs, WCA_specs, MI_specs, HDI_specs, NNI_specs, MRI_specs, MICE_specs]
#
#     FDM.show_specs_distributions(All_specs, All_names, off_set=Regular_specs, missing_percentage=missing_percentage)
