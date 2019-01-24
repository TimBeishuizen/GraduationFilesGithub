import csv
import matplotlib.pyplot as plt
import numpy as np

from Testing_TPOT.AverageTPOTResultMethod import average_TPOT_results

rows = []

# data_file = 'TPOT_results.csv'
data_file = 'TPOT_results_new.csv'

with open(data_file, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        rows.append(row)

output_values = []
output_specs = []
output_legend = []

for row in rows:
    output_values.append(np.asarray(row[2:]).astype(float).tolist())
    output_spec = []

    if row[0][0:2] == 'MO':
        output_spec.append('MicroOrganisms')
    elif row[0][0:2] == 'AR':
        output_spec.append('Arcene')
    elif row[0][0:2] == 'RS':
        output_spec.append('RSCTC')
    elif row[0][0:2] == 'PS':
        output_spec.append('Psoriasis')
    else:
        raise ValueError("Unknown dataset")

    if row[0][3:5] == 'rs':
        output_spec.append('Regular selection')
    elif row[0][3:5] == 'fs':
        output_spec.append('Feature selection')
    else:
        raise ValueError("Unknown selection type")

    if row[0][6:8] == 'ra':
        output_spec.append('Regular algorithms')
    elif row[0][6:8] == 'fa':
        output_spec.append('FS algorithms')
    else:
        raise ValueError("Unknown algorithm type")

    output_spec.append(row[1])

    output_legend.append(' '.join(output_spec))

    output_specs.append(output_spec)

final_legend = []

avg_values, avg_specs = average_TPOT_results(output_values, output_specs)

colors = ['c-', 'm-', 'g-', 'y-']

avg_legend = []

cur_data = ['MicroOrganisms', 'Arcene'] #['RSCTC', 'Psoriasis']# #,

markers = ['-', '-', '-', '-', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '-.', '-.', '-.', '-.']
color_markers = ['c', 'r', 'g', 'y', 'c', 'r', 'g', 'y', 'c', 'r', 'g', 'y', 'c', 'r', 'g', 'y']

for i in range(len(avg_values)):
    print(i)
    if avg_specs[i][0] in cur_data:
        plt.step(range(121), avg_values[i], color=color_markers[i], lineStyle=markers[i])
        avg_legend.append(' '.join(avg_specs[i]))
        print(avg_legend[-1])
        print(avg_values[i][-1])
plt.legend(avg_legend)
plt.xlabel('Time (min)')
plt.ylabel('FS_accuracy ([0, 1)')
plt.title('The average TPOT performance for the %s dataset'  %cur_data)
plt.show()



for i in range(int(len(output_values)/2)):
    if output_specs[i * 2][0] in cur_data:
        plt.step([0] + output_values[i*2], [0] + output_values[i*2 + 1], colors[int((i%20)/5)])
        final_legend.append(output_legend[i * 2])

plt.legend(final_legend)
# plt.show()

