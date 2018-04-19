import csv
import numpy as np
import os

path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\MulticollinearityData'

CSV_folder = os.path.join(path, 'CSV_folder')
corr_70_folder = os.path.join(path, 'Correlation_70_folder')
corr_95_folder = os.path.join(path, 'Correlation_95_folder')

# Collect traces for correlation coefficient
print("Collecting traces for correlation coefficient...")

check_feat = 54675

trace_cov_mat = np.zeros((check_feat, 1))

# Going through all files first
for i in range(check_feat):
    with open(os.path.join(CSV_folder, 'cov_mat_%i.csv' % i), 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        cov_mat = np.asarray(list(csv_reader), dtype=float)

    trace_cov_mat[i, 0] = cov_mat[0, i]

    if i % np.ceil(check_feat/10) == 0:
        print("Currently at feature %i of %i" % (i+1, check_feat))

# Finding the actual values
print("Collect covariance coefficients...")

corr_95 = 0.
corr_70 = 0.

for i in range(check_feat):
    with open(os.path.join(CSV_folder, 'cov_mat_%i.csv' % i), 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        cov_mat = np.asarray(list(csv_reader), dtype=float)

    feat_corr_70 = 0
    feat_corr_95 = 0

    # Start correlation finding
    # print("Start correlation finding for feature %i" % i)

    list_corr_70 = []
    list_corr_95 = []

    for j in range(check_feat):
        # Compute correlation
        corr_value = cov_mat[0, j] / np.sqrt(trace_cov_mat[i, 0] * trace_cov_mat[j, 0])

        if corr_value > 0.70 and i != j:
            feat_corr_70 += 1
            list_corr_70.append(j)
            if corr_value > 0.95:
                feat_corr_95 += 1
                list_corr_95.append(j)

    corr_95 += feat_corr_95 / 2
    corr_70 += feat_corr_70 / 2

    if i % np.ceil(check_feat/10) == 0:
        print("Currently at feature %i of %i" % (i+1, check_feat))

    with open(os.path.join(corr_70_folder, 'corr_feature_%i.csv' % i), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(list_corr_70)

    with open(os.path.join(corr_95_folder, 'corr_feature_%i.csv' % i), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(list_corr_95)

perc_corr_95 = corr_95 / ((check_feat * check_feat) / 2 - check_feat)
perc_corr_70 = corr_70 / ((check_feat * check_feat) / 2 - check_feat)

print("Part of correlations bigger than 0.95 is %f" % perc_corr_95)
print("Part of correlations bigger than 0.70 is %f" % perc_corr_70)
