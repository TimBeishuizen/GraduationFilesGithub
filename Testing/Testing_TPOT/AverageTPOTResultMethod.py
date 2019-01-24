

datasets = ['MicroOrganisms', 'Arcene', 'RSCTC', 'Psoriasis']
selections = ['Regular selection', 'Feature selection']
algorithms = ['Regular algorithms', 'FS algorithms']

def average_TPOT_results(output_values, output_specs):

    output_specs_new = []
    output_values_new = []
    output_time = range(121)

    for dataset in datasets:
        for selection in selections:
            for algorithm in algorithms:
                output_specs_new.append([dataset, selection, algorithm])

                acc_values = [0] * len(output_time)

                for i in range(int(len(output_values)/2)):
                    if output_specs[i*2][0] != dataset or \
                       output_specs[i*2][1] != selection or \
                       output_specs[i*2][2] != algorithm:
                        continue

                    for j in range(len(output_values[2*i]) - 1):
                        acc_values[int(output_values[2*i][j])] += \
                            (output_values[2 * i + 1][j + 1] - output_values[2 * i + 1][j]) / 5

                for j in range(1, len(acc_values)):
                    acc_values[j] += acc_values[j - 1]

                output_values_new.append(acc_values)

    return output_values_new, output_specs_new
