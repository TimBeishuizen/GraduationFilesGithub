import numpy as np

def testing_gene_relations(gene_ids, gene_set, printing=False):
    """ Testing the relations between a set of genes

    :param gene_ids: The ids of the tested genes
    :param gene_set: The details of the tested genes
    :param printing: Printing additional data, default = False
    :return: relations
    """

    # Extract all prcess relations
    process_relations = []
    for name in gene_ids:
        process_relations.append(gene_set[name]['Process Relations'])

    sign_process = {}

    for i in range(len(gene_ids)):
        if i%int(len(gene_ids)/20)==0:
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

    print("%i processes have a relation with multiple genes" % len(sign_process))
    relations = 0
    most_process = 'None'
    for process in sign_process:
        if len(sign_process[process]['Genes']) > relations:
            relations = len(sign_process[process]['Genes'])
            most_process = sign_process[process]['Description']

    print("Maximum number of genes in relation with one process is %i for process '%s'" % (relations, most_process))

    # Print results for the relations
    for process in sign_process:
        gene_number = len(sign_process[process]['Genes'])
        gene_description = sign_process[process]['Description']
        if gene_number > relations / 4:
            print("%i relations for process '%s' " % (gene_number, gene_description))

    return sign_process

def remove_double_genes(sample_values, gene_ids, removable_gene_ids):
    """ Removes all values and genes with the removable gene ids.

    :param sample_values: Values of the genes
    :param gene_ids: The id of the genes
    :param removable_gene_ids: The id of the genes to be removed
    :return: sample values and gene ids of the remaining values
    """

    # Remove values both in non-lesional and lesional
    print("Removing double values")
    for gene in removable_gene_ids:
        if gene in gene_ids:
            index = gene_ids.index(gene)
            gene_ids.remove(gene)
            sample_values = np.delete(sample_values, index, axis=0)

    return sample_values, gene_ids
