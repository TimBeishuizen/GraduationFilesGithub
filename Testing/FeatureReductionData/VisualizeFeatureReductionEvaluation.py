import numpy as np
import csv
import matplotlib.pyplot as PLT

data_name = 'Psoriasis'

values = []

with open('Filter_T_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        values.append(row)

extra_MI_values = []

with open('Filter_MI_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        extra_MI_values.append(row)

forward_backward_values = []

with open('Wrapper_Forward_Backward_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        forward_backward_values.append(row)

PTA_values = []

with open('PTA_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        PTA_values.append(row)

Floating_values = []

with open('Floating_values.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        Floating_values.append(row)

values = np.asarray(values)
extra_MI_values = np.asarray(extra_MI_values)
forward_backward_values = np.asarray(forward_backward_values)
PTA_values = np.asarray(PTA_values)
Floating_values = np.asarray(Floating_values)

print(forward_backward_values.shape)
print(PTA_values.shape)
print(Floating_values.shape)

filter_values = np.append(values, extra_MI_values, axis = 1)
wrapper_values = np.concatenate((forward_backward_values, PTA_values, Floating_values), axis=1)

plot_legend = []


def return_label(rank, threshold):
    label = ''
    if rank in ['T-test', 'Random'] and threshold in ['0.05', '0.001']:
        label = label + 'o'
    elif rank in ['T-test', 'Random'] and threshold == '0.01':
        label = label + 's'
    elif rank == 'MI' and threshold in ['0.1', '0.001']:
        label = label + 'x'
    elif rank == 'MI' and threshold in ['0.2', '0.01']:
        label = label + 'd'
    else:
        raise ValueError('Unknown rank and threshold combination')
    return label


def compute_fraction(data_name, number):
    if data_name == 'Psoriasis':
        return number / 54676
    elif data_name == 'RSCTC':
        return number / 54676
    elif data_name == 'Arcene':
        return number / 10000
    elif data_name == 'MicroOrganisms':
        return number / 1300
    else:
        raise ValueError("Unknown data label")

# # Plotting filter values
# for i in range(filter_values.shape[1]):
#     if filter_values[0, i] == 'MicroOrganisms':
#         style_label = 'r' + return_label(filter_values[1, i], filter_values[2, i])
#
#         legend_label = 'Filter: ' + filter_values[1, i] + ', ' + filter_values[2, i]
#         val_score = float(filter_values[4, i])
#         fraction = compute_fraction(filter_values[0, i], int(filter_values[5, i]))
#         PLT.plot(fraction, val_score, style_label, label=legend_label)

# Plotting Wrapper values
for i in range(wrapper_values.shape[1]):
    if wrapper_values[0, i] == 'MicroOrganisms':
        if wrapper_values[1, i] == 'forward':
            style_label = 'b' + return_label(wrapper_values[3, i], wrapper_values[2, i])
            legend_label = wrapper_values[1, i] + ': order = ' + wrapper_values[3, i] + ', ' + \
                           wrapper_values[2, i]

        elif wrapper_values[1, i] == 'backward':
            style_label = 'y' + return_label(wrapper_values[3, i], wrapper_values[2, i])
            legend_label = wrapper_values[1, i] + ': order = ' + wrapper_values[3, i] + ', ' + \
                           wrapper_values[2, i]

        elif wrapper_values[1, i][0:2] == '[2':
            style_label = 'g' + return_label(wrapper_values[3, i], wrapper_values[2, i])
            legend_label = 'PTA, [l, r] = ' + wrapper_values[1, i] + ': order = ' + wrapper_values[3, i] + ', ' + wrapper_values[2, i]
        elif wrapper_values[1, i][0:2] == '[5':
            style_label = 'c' + return_label(wrapper_values[3, i], wrapper_values[2, i])
            legend_label = 'PTA, [l, r] = ' + wrapper_values[1, i] + ': order = ' + wrapper_values[3, i] + ', ' + \
                           wrapper_values[2, i]
        elif wrapper_values[1, i] == '1':
            style_label = 'm' + return_label(wrapper_values[3, i], wrapper_values[2, i])
            legend_label = 'Floating search: order = ' + wrapper_values[3, i] + ', ' + \
                           wrapper_values[2, i]
        else:
            raise ValueError

        val_score = float(wrapper_values[5, i])
        fraction = compute_fraction(wrapper_values[0, i], int(wrapper_values[6, i]))
        PLT.plot(fraction, val_score, style_label, label=legend_label)

#PLT.axis([0, 1, 0, 1])
PLT.axis([0, 0.2, 0.4, 1])
PLT.xlabel("The feature fraction")
PLT.ylabel("The test score")
PLT.title("The spectrum for all sequential wrapper methods")
PLT.legend(loc='center left', bbox_to_anchor=(0.7, 0.3))
PLT.show()




