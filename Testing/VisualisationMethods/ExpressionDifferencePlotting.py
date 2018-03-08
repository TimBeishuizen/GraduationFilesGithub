import math
import matplotlib.pyplot as plt


def plot_multiple_processes(processes, process_genes, gene_ids, values_1, values_2, sign_values=None, frame_size=4):
    """ Plots multiple processes in several plots

    :param processes: the processes to be plotted
    :param process_genes: the genes of the processes
    :param gene_ids: the ids of the genes
    :param values_1: the values of group 1, needs to be of identical length of the values of group 2
    :param values_2: the values of group 2, needs to be of identical length of the values of group 1
    :param sign_values: the significance values of the processes. If None (default), the are not used
    :param frame_size: the frame size of the plot window
    :return: shown plots
    """

    # For number of plot range
    for p in range(int(math.ceil(len(processes) / (frame_size * frame_size)))):
        f, plots = plt.subplots(frame_size, frame_size)
        f.text(0.5, 0.06, 'non-lesional skin (avg expression)', ha='center')
        f.text(0.1, 0.5, 'lesional skin (avg expression)', va='center', rotation='vertical')

        # For subplot range within plot
        for i in range(frame_size * frame_size):

            # Initialize values
            group_2_process = []
            group_1_process = []
            group_2_not_process = []
            group_1_not_process = []

            # Check if plot can be made
            if p * (frame_size * frame_size) + i < len(processes):
                process = processes[p * (frame_size * frame_size) + i]

                # Add values to their respective lists
                for j in range(len(gene_ids)):
                    if gene_ids[j] in process_genes[process]:
                        group_2_process.append(values_2[j])
                        group_1_process.append(values_1[j])
                    else:
                        group_2_not_process.append(values_2[j])
                        group_1_not_process.append(values_1[j])

                    ax = plots[i % frame_size, math.floor(i / frame_size)]

                # Plot the values
                ax.scatter(group_2_not_process, group_1_not_process)
                ax.scatter(group_2_process, group_1_process)

                for item in ([ax.title] +
                              ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(8)

                # Add significance values to the plots
                if sign_values is None:
                    plots[i % frame_size, math.floor(i / frame_size)].set_title("%s" % process, fontsize=8)
                else:
                    sign_value = sign_values[p * (frame_size * frame_size) + i]
                    plots[i % frame_size, math.floor(i / frame_size)].set_title("%s, (%f)" % (process, sign_value), fontsize=8)

        plt.show()


def plot_gene_set(gene_ids, spec_gene_ids, specific_name, values_1, values_2):
    """ Plot specific genes from a set of genes

    :param gene_ids: All gene ids
    :param spec_gene_ids: specific gene ids
    :param specific_name: The name of the specific gene set
    :param values_1: The values of group 1
    :param values_2: The values of group 2
    :return: a plot of the genes
    """

    group_1_spec = []
    group_2_spec = []
    group_1_norm = []
    group_2_norm = []

    for i in range(len(gene_ids)):
        if gene_ids[i] in spec_gene_ids:
            group_1_spec.append(values_1[i])
            group_2_spec.append(values_2[i])
        else:
            group_1_norm.append(values_1[i])
            group_2_norm.append(values_2[i])

    plt.scatter(group_2_norm, group_1_norm)
    plt.scatter(group_2_spec, group_1_spec)
    plt.plot([-1.5, 6], [-1.5, 6])
    plt.title("Genes involved in %s" % specific_name)
    plt.legend(["Base line", "Significant genes", "%s genes" % specific_name])

    plt.show()
