import csv
import numpy as np

old_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\HepatitisSet\dataset_55_hepatitis.csv'
new_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\HepatitisSet\HepatitisDataset.csv'

matrix = []

# Opening CSV file
with open(old_path, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        matrix.append(row)

for i in range(1, len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i][j] == 'male':
            matrix[i][j] = True
        elif matrix[i][j] == 'female':
            matrix[i][j] = False
        elif matrix[i][j] == '?':
            matrix[i][j] = None
        elif matrix[i][j] == 'yes':
            matrix[i][j] = True
        elif matrix[i][j] == 'no':
            matrix[i][j] = False
        elif matrix[i][j] == 'LIVE':
            matrix[i][j] = 0
        elif matrix[i][j] == 'DIE':
            matrix[i][j] = 1
        else:
            try:
                matrix[i][j] = int(matrix[i][j])
            except ValueError:
                matrix[i][j] = float(matrix[i][j])

# Opening CSV file
with open(new_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(matrix)