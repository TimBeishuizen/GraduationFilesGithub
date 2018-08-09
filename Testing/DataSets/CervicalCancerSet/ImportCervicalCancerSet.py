import csv
import numpy as np
import math

old_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\CervicalCancerSet\risk_factors_cervical_cancer.csv'
new_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\CervicalCancerSet\CervicalCancerData.csv'

matrix = []

# Opening CSV file
with open(old_path, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        matrix.append(row)

missing = 0

for i in range(1, len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i][j] == '?':
            matrix[i][j] = None
            missing +=1
        elif j not in [1, 2, 3, 5, 6, 8, 10, 12, 25, 26, 27, 32, 33, 34, 35]:
            if float(matrix[i][j]) == 0:
                matrix[i][j] = False
            elif float(matrix[i][j]) == 1:
                matrix[i][j] = True
        else:
            try:
                matrix[i][j] = int(matrix[i][j])
            except ValueError:
                if float(matrix[i][j]) % 1 == .0:
                    matrix[i][j] = int(math.floor(float(matrix[i][j])))
                else:
                    matrix[i][j] = float(matrix[i][j])

print("Data set size is %i by %i" % (len(matrix), len(matrix[0])))

for j in range(len(matrix[0])):
    attribute = []
    missing_2 = 0
    for i in range(1, len(matrix)):

        if matrix[i][j] == None:
            missing_2 += 1
        else:
            attribute.append(matrix[i][j])
    print("Attribute %i with name %s and %i missing values"  % (j, matrix[0][j], missing_2))
    #print(set(attribute))

for j in range(28, len(matrix[0])):
    attribute = []
    true = [0] * len(matrix[0])
    false = [0] * len(matrix[0])
    missing_3 = [0] * len(matrix[0])
    for i in range(1, len(matrix)):
        if matrix[i][j] == None:
            missing_3[j] = missing_3[j] + 1
        elif matrix[i][j]:
            true[j] = true[j] + 1
        elif not matrix[i][j]:
            false[j] = false[j] + 1
        else:
            raise ValueError

    print("Attribute %i with name %s has %i true, %i false and %i missing values" % (j, matrix[0][j], true[j], false[j], missing_3[j]))

matrix[0] = matrix[0][:28] + ['Diagnostics']

for i in range(1, len(matrix)):
    y_val = matrix[i][32:]
    matrix[i] = matrix[i][:28]
    matrix[i].append(sum(y_val))



print("Missing %i values, %f of total values" %(missing, missing/(len(matrix) * len(matrix[0]))))

# Opening CSV file
with open(new_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(matrix)