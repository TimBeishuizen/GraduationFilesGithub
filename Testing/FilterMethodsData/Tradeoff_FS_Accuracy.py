import numpy as np
import csv
import matplotlib.pyplot as plt

used_data = 'MO'
used_test = 'T'
data_type = 'val'

val_complete = []
feat_complete = []

with open(used_data + '_' + used_test + '_feat_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        feat_complete.append(row)

with open(used_data + '_' + used_test + '_val_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        val_complete.append(row)

val_complete = np.array(val_complete, dtype=float)
feat_complete = np.array(feat_complete, dtype=int)

plt.plot(feat_complete[1, 1:], val_complete[1, 1:])

#exp = range(1000)

outcome = []

for i in feat_complete[1, 1:]:
    outcome.append(0.99 ** i)

final_val = val_complete[1, 1:] * outcome

print(final_val)

plt.plot(feat_complete[1, 1:], outcome, feat_complete[1, 1:], final_val)
plt.title("The impact of a correction factor on an exemplary filter method")
plt.xlabel("The number of features")
plt.ylabel("The accuracy/correction factor")
plt.legend(['accuracy filter method', 'correction factor', 'FS_accuracy filter method'])
plt.axis([0, 300, 0, 1])
#plt.xscale("log")
plt.show()

plt.show()