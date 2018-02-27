import numpy as np


def testing_gene_relations(gene_ids, gene_set, printing=0, relation_type = "Process"):
    """ Testing the relations between a set of genes

    :param gene_ids: The ids of the tested genes
    :param gene_set: The details of the tested genes
    :param printing: Printing additional data, default = 0
    :param relation_type: The relation type that is checked. This can be "Process" (default), "Cellular" and "Molecular"
    :return: relations
    """

    # Test whether relation type is right
    if relation_type not in ["Process", "Cellular", "Molecular"]:
        raise ValueError("Relation type %s is not available" % relation_type)

    # Extract all prcess relations
    process_relations = []
    for name in gene_ids:
        process_relations.append(gene_set[name]['%s Relations' % relation_type])

    sign_process = {}

    for i in range(len(gene_ids)):
        if len(gene_ids) > 20 and i%int(len(gene_ids)/20)==0:
            print("Currently at gene %i" %i)
        for j in range(i + 1, len(gene_ids)):
            for process in process_relations[i]:
                if process in process_relations[j]:
                    if len(process) < 2:
                        continue
                    elif process[0] not in sign_process:

                        # print("Process '%s' has multiple relations to this gene set" % (process[1]))
                        sign_process[process[0]] = {'ID': process[0], 'Description': process[1], 'Genes': set()}
                        if len(process) > 2:
                            sign_process[process[0]]['Acquired'] = process[2]
                        else:
                            sign_process[process[0]]['Acquired'] = 'Unknown'

                    sign_process[process[0]]['Genes'].add(gene_ids[i])
                    sign_process[process[0]]['Genes'].add(gene_ids[j])

    print("%i %s aspects have a relation with multiple genes" % (len(sign_process), relation_type))
    relations = 0
    most_process = 'None'
    for process in sign_process:
        if len(sign_process[process]['Genes']) > relations:
            relations = len(sign_process[process]['Genes'])
            most_process = sign_process[process]['Description']

    print("Maximum number of genes in relation with one %s aspect is %i for %s aspect '%s'" % (relation_type, relations,
                                                                                               relation_type, most_process))

    gene_number = []
    gene_description = []

    # Print results for the relations
    for process in sign_process:
        gene_number.append(len(sign_process[process]['Genes']))
        gene_description.append(sign_process[process]['Description'])

    sorting = np.argsort(gene_number)

    for sort_index in np.flip(sorting, 0).tolist():
        if printing >= 1 and gene_number[sort_index] / printing > 1:
            print("%i relations for process '%s' " % (gene_number[sort_index], gene_description[sort_index]))
        elif 0 <printing < 1 and gene_number[sort_index] > relations * printing:
            print("%i relations for process '%s' " % (gene_number[sort_index], gene_description[sort_index]))

    return sign_process


def remove_double_genes(sample_values, gene_ids, removable_gene_ids):
    """ Removes all values and genes with the removable gene ids.

    :param sample_values: Values of the genes
    :param gene_ids: The id of the genes
    :param removable_gene_ids: The id of the genes to be removed
    :return: sample values and gene ids of the remaining values
    """

    # Remove values both in non-lesional and lesional
    for gene in removable_gene_ids:
        if gene in gene_ids:
            index = gene_ids.index(gene)
            gene_ids.remove(gene)
            sample_values = np.delete(sample_values, index, axis=0)

    print("%i genes left after removing double genes" % len(gene_ids))
    return sample_values, gene_ids


def remove_insignificant_processes(processes, process_genes, gene_ids, values_1, values_2, gene_threshold=3,
                                   sign_threshold=0.7, ordering=True):
    """ Removes processes that have no significant presence

    :param processes: The processes to be checked
    :param process_genes: The genes for every process
    :param gene_ids: The id of the genes to be cheched
    :param values_1: The values from the first group. Should be a list with same length of values_2
    :param values_2: The values from the second group. Should be a list with same length of values_1
    :param gene_threshold: The lower boundary of genes the process has to be related with
    :param sign_threshold: The threshold for significant presence. THe higher the threshold the more the process has to
        differ between group 1 values and group 2 values
    :param ordering: Whether the processes should be ordered by significance, default = True
    :return: A pruned list with processes and their significance values
    """
    sign_values = []

    process_removal = processes.copy()
    for process in process_removal:
        if len(process_genes[process]) < gene_threshold:
            processes.remove(process)
            continue

        sign_value = 0
        for j in range(len(gene_ids)):
            if gene_ids[j] in process_genes[process]:
                sign_value += abs(values_1[j] - values_2[j])
        sign_value /= len(process_genes[process])
        if sign_value < sign_threshold:
            processes.remove(process)
        else:
            sign_values.append(sign_value)

    print("%i are left" % (len(processes)))

    if ordering:
        print("Ordering after removal...")
        order = np.argsort(np.array(sign_values), axis=0)
        sign_values = [sign_values[i] for i in reversed(order)]
        processes = [processes[i] for i in reversed(order)]

    return processes, sign_values

