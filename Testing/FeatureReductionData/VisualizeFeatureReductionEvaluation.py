import numpy as np
import csv
import matplotlib.pyplot as PLT

data_name = 'Psoriasis'

filter_values = []

# with open('Filter_T_values.csv', 'r', newline='') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     for row in csv_reader:
#         filter_values.append(row)
#
# filter_values = np.asarray(filter_values)

wrapper_values = []

if data_name in ['MO', 'Arcene']:
    with open(data_name + '_Wrapper_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            wrapper_values.append(row)

    wrapper_values = np.asarray(wrapper_values)

elif data_name in ['Psoriasis', 'RSCTC']:
    quick_values = []
    with open(data_name + '_Wrapper_Quick_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            quick_values.append(row)

    float_values = []
    with open(data_name + '_Wrapper_Float_values.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            float_values.append(row)

    wrapper_values = np.concatenate((quick_values, float_values), axis=1)

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
        if wrapper_values[1, i] == 'forward':
            style_label = 'b' + return_label(wrapper_values[3, i], wrapper_values[2, i])
            legend_label = wrapper_values[1, i] + ': order = ' + wrapper_values[3, i] + ', ' + \
                           wrapper_values[2, i]

        elif wrapper_values[1, i] == 'backward':
            continue
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
        elif wrapper_values[1, i] == 'floating':
            style_label = 'm' + return_label(wrapper_values[3, i], wrapper_values[2, i])
            legend_label = 'Floating search: order = ' + wrapper_values[3, i] + ', ' + \
                           wrapper_values[2, i]
        else:
            raise ValueError

        val_score = float(wrapper_values[10, i])
        fraction = compute_fraction(wrapper_values[0, i], int(wrapper_values[6, i]))
        PLT.plot(fraction, val_score, style_label, label=legend_label)

# PLT.axis([0, 1, 0, 1])
# PLT.axis([0, 0.2, 0.4, 1])
PLT.xlabel("The feature fraction")
PLT.ylabel("The F1 score")
PLT.title("The spectrum for all sequential wrapper methods for the %s dataset" % (data_name))
PLT.legend(loc='center left', bbox_to_anchor=(0.7, 0.3))
PLT.show()




