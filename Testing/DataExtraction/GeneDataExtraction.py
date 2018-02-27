import numpy as np
import os


def read_gene_data():
    """ Reads the gene - data from the corresponding data set

       :return: The gene data
       """

    # Changes directory to data directory
    owd = os.getcwd()
    os.chdir('../DataSets/SkinDiseaseDataSet')

    # Opens data file
    file = open('GPL570-55999.txt', encoding='utf-8')

    set_data = []
    gene_data = []

    for line in file:

        # Set data
        if line[0] == '#':
            set_data.append(line)

        elif line[0:2] == 'ID':
            gene_data_spec = line

        # Gene data
        else:
            gene_data.append(line)

    # return to working directory
    os.chdir(owd)

    return set_data, gene_data_spec, gene_data


def extract_set_data(set_data):
    """ Returns the details of the data set

    :param set_data: All information about the data set
    :return: Details of the information in a set form
    """

    # Dictionary for the series
    set_details = {}

    # Add words for the dictionary
    for line in set_data:
        line = line[1:-1]
        line_data = line.split(' ', 1)
        set_details[line_data[0]] = line_data[1]

    return set_details


def extract_gene_spec(gene_data_spec):
    """ Returns the names of the specifications of the gene matrix

    :param gene_data_spec: An unprocessed specification string
    :return: A properly processed specification string
    """

    gene_data_spec = gene_data_spec[0:-1]
    gene_spec = gene_data_spec.split('\t')

    return gene_spec


def extract_gene_details(gene_data):
    """ Extract all details of the genes

    :param gene_data: The gene matrix with all the data
    :return: A processed matrix of the data
    """

    genes = []

    for line in gene_data:
        line = line[0:-1]
        line_data = line.split('\t')
        genes.append(line_data)

    return genes


def extract_gene_relations(gene_relations):
    """ Extracts the relations present in the genes

    :param gene_relations: All processes the gene is found a relation with
    :return:
    """

    # Split the text in useful separations
    topic_separation = gene_relations.split(' /// ')
    detailed_separation = [topic.split(' // ') for topic in topic_separation]

    return detailed_separation


def extract_gene_data():
    """ Extracts all relevant gene data from the gene data set

    :return: the values in the data sets with their corresponding names and details
    """

    # Read and extract all viable data
    set_data, gene_data_spec, gene_data = read_gene_data()
    # set_details = extract_set_data(set_data)
    gene_spec = extract_gene_spec(gene_data_spec)
    genes = extract_gene_details(gene_data)

    # Create a set consisting of every gene
    gene_set = {}

    # Make a set for every gene
    for gene in genes:
        gene_details = {}
        for i in range(len(gene_spec)):
            if gene_spec[i] == "Gene Ontology Biological Process":

                # Find the relations of the gene
                gene_details['Process Relations'] = extract_gene_relations(gene[i])
                continue
            elif gene_spec[i] == "Gene Ontology Cellular Component":

                # Find the relations of the gene
                gene_details['Cellular Relations'] = extract_gene_relations(gene[i])
                continue
            elif gene_spec[i] == "Gene Ontology Molecular Function":

                # Find the relations of the gene
                gene_details['Molecular Relations'] = extract_gene_relations(gene[i])
                continue
            gene_details[gene_spec[i]] = gene[i]

        # Add the set to the gene pool
        gene_set[gene[0]] = gene_details

    return gene_set