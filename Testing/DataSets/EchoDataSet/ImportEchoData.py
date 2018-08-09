import csv
import numpy as np

old_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\EchoDataSet\dataset_2208_echoMonths.csv'
new_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\EchoDataSet\EchoDataset.csv'

matrix = []

# Opening CSV file
with open(old_path, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        matrix.append(row)

for i in range(1, len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i][j] == '?':
            matrix[i][j] = None
        elif j in [0, 2, 8] and matrix[i][j] == '0':
            matrix[i][j] = False
        elif j in [0, 2, 8] and matrix[i][j] == '1':
            matrix[i][j] = True
        else:
            try:
                matrix[i][j] = int(matrix[i][j])
            except ValueError:
                matrix[i][j] = float(matrix[i][j])

# Opening CSV file
with open(new_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(matrix)