def extract_processes(gene_ids, gene_set, relation_type="Process"):
    """ Extracts all relation_type aspects the genes are involved with

    :param gene_ids: the id of the genes to be checked
    :param gene_set: The set with all gene information
    :param relation_type: The type of relation. Three possible typesL: "Process" (default), "Cellular", "Molecular"
    :return: the processes, and the genes related to that process
    """

    processes = []
    process_genes = {}

    for gene in gene_ids:
        for process_relation in gene_set[gene]['%s Relations' % relation_type]:
            if len(process_relation) > 1:
                if process_relation[1] not in processes:
                    processes.append(process_relation[1])
                    process_genes[process_relation[1]] = []
                else:
                    process_genes[process_relation[1]].append(gene)
    print("%i %s aspects are involved" % (len(processes), relation_type))

    return processes, process_genes