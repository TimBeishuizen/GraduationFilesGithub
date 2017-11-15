from FileExtraction import ExtractFiles as EF
import random
import numpy as np


def preprocess_gene_file(gene_strings):
    """ Preporcessing the gene file

    :param gene_strings: A string with the gene data
    :return: Preprocessed genes
    """

    # Create lists for the strands and the names of the string
    gene_strands = []
    gene_names = []

    # While there are still genes available, add them
    last = gene_strings.find('>', 1, 200)

    while last != -1:
        # Find the name location
        gene_string = gene_strings[0:last - 2]
        name_end = gene_string.find('\n')

        # Append the name and the strand
        gene_names.append(gene_string[1:name_end])
        gene_strands.append(gene_string[name_end + 2:])

        # Cut the latest gene from the string
        gene_strings = gene_strings[last:]
        last = gene_strings.find('>', 1, 200)

    # Write genes such that ever char is one value in the list
    for i in range(len(gene_strands)):
        gene_strands[i] = list(gene_strands[i])

    # Make it a numpy matrix, which are easier to work with
    gene_strands = np.mat(gene_strands)

    gene_options = [['A'], ['G'], ['T'], ['C']]
    mask = np.isin(gene_strands, gene_options)

    inv_genes = np.array([y for y in np.where(mask == False) if 0 not in y.shape])

    if inv_genes != []:
        inv_loc = -np.sort(-np.unique(inv_genes[0,:]))
        for loc in inv_loc:
            gene_strands = np.delete(gene_strands, loc, 0)

    return gene_strands, gene_names


def create_combination_gene_file(data_type, gene_type):
    """ Creates a file of genes with data_type and gene_type with the true and false values randomly mixed

    :param data_type: data_type: Defines the type for machine learning: train or test
    :param gene_type: Defines the type of gene: acceptor or donor 
    :return: A generated gene list file with both True and False
    """

    # Read the strings
    true_strings = EF.read_gene_file(data_type, True, gene_type)
    false_strings = EF.read_gene_file(data_type, False, gene_type)

    # Preprocess the strings
    true_strands, true_names = preprocess_gene_file(true_strings)
    false_strands, false_names = preprocess_gene_file(false_strings)

    # Outcomes of the strings
    true_outcome = [True] * len(true_strands)
    false_outcome = [False] * len(false_strands)

    # Combine lists
    strands = np.append(true_strands, false_strands, axis=0)
    names = true_names
    names.extend(false_names)
    outcome = true_outcome
    outcome.extend(false_outcome)

    # Create an order to shuffle
    range_list = [x for x in range(len(strands))]
    random.shuffle(range_list)

    # Shuffle the lists
    rand_strands = np.squeeze(np.array([strands[i] for i in range_list]))
    rand_names = [names[i] for i in range_list]
    rand_outcome = [outcome[i] for i in range_list]



    return rand_strands, rand_names, rand_outcome


