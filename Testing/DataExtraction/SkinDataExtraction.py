import os
import re
import numpy as np


def read_data(name):
    """ Reads the data from a data set

    :param name: The name of the data set
    :return: The read data set
    """

    # Changes directory to data directory
    owd = os.getcwd()
    os.chdir('../DataSets/SkinDiseaseDataSet')

    # Opens data file
    file = open(name + '_series_matrix.txt', encoding='utf-8')

    # Creates locations for the data
    series_data = []
    sample_data = []
    sample_matrix = []

    line = file.readline()

    n = 0

    while line[0:24] != "!series_matrix_table_end":

        # Series data
        if line[0:7] == "!Series":
            series_data.append(line)

          # Sample data
        elif line[0:7] == "!Sample":
            sample_data.append(line)

        # Matrix data
        elif line[0] == '"':
            sample_matrix.append(line)

        line = file.readline()

    # return to working directory
    os.chdir(owd)

    return series_data, sample_data, sample_matrix


def extract_series_data(series_data):
    """ Extracts the series data

    :param series_data: The data of this specific series
    :return: The series data structured
    """

    # Dictionary for the series
    series = {}

    # Add words for the dictionary
    for line in series_data:
        line = line[8:-2]
        line_data = line.split('\t"', 1)
        series[line_data[0]] = line_data[1]

    return series


def extract_sample_data(sample_data):
    """ Extract the data from the samples

    :param sample_data: The sampled data
    :return: The data extracted into functional usable data
    """

    # Samples for the series with each sample having its own dictionary
    samples = []

    # Create a dictionary for each sample
    first_line = sample_data[0][8:-2]
    first_line = re.sub('"', '', first_line)
    line_data = first_line.split('\t')
    line_attrib = line_data[0]
    for line_sample_data in line_data[1:]:
        samples.append({line_attrib: line_sample_data})

    # Add words for each sample
    for line in sample_data[1:]:
        proc_line =  re.sub('"', '', line[8:-2])
        line_data = proc_line.split('\t')
        line_attrib = line_data[0]
        for i in range(len(line_data[1:])):
            samples[i][line_attrib] = line_data[i + 1]

    return samples


def extract_sample_matrix(sample_matrix):
    """ Extracts the data for the sample matrix

    :param sample_matrix: The sample matrix values
    :return: A matrix with the sample values
    """

    n_genes = len(sample_matrix)

    # Extract the sample_ids
    first_line = re.sub('"','', sample_matrix[0][:-2])
    sample_ids = first_line.split('\t')[1:]

    # Extract the gene_ids
    gene_ids = []
    matrix_values = np.zeros((n_genes-1, len(sample_ids)))

    for i in range(len(sample_matrix[1:])):
        # Remove quotation marks and a possible enter at the end
        line_data = re.sub('"', '', sample_matrix[i + 1])
        if line_data[-2:] == '\n':
            line_data = line_data[:-2]

        # Split per tab
        line_values = line_data.split('\t')
        gene_ids.append(line_values[0])
        matrix_values[i, :] = line_values[1:]

    # Make the matrix consist of floats
    matrix_values.astype(float)

    return sample_ids, gene_ids, matrix_values.transpose()


def encode_output(skin_types, disease_type):
    """ Encodes the skin types

    :param skin_types:
    :param disease_type: The type of disease (psoriasis or AD (atopic dermatitis)
    :return: An encoding of the skin types
    """

    if disease_type == 'psoriasis':
        encoding = np.zeros((skin_types.shape[0]))
    elif disease_type == 'AD':
        encoding = np.zeros((skin_types.shape[0]))
    else:
        raise NotImplementedError("This disease is not implemented")

    for i in range(skin_types.shape[0]):
        if skin_types[i] in ['NN', 'Normal', 'PRE, ANL']:
            encoding[i] = 0
        elif disease_type == 'psoriasis' and skin_types[i] in ['NL', 'PN', 'non-lesional', 'NS', 'NLS'] or (disease_type == 'AD' and skin_types[i] in ['ANL', 'NLS', 'NL', 'POST, ANL']):
            encoding[i] = 1
        elif (disease_type == 'psoriasis' and skin_types[i] in ['PP', 'lesional', 'LS', 'Mild psoriasis']) or (disease_type == 'AD' and skin_types[i] in ['ALS', 'AL', 'PRE, AL']):
            encoding[i] = 2
        elif (disease_type == 'psoriasis' and skin_types[i] in ['Severe psoriasis']) or (disease_type == 'AD' and skin_types[i] in ['CLS', 'POST, AL']):
            encoding[i] = 3
        else:
            raise NotImplementedError("The label " + skin_types[i] + " is not implemented")

    return encoding


def extract_data(name):
    """ Extracts all relevant data from the data set with the name name

    :param name: The name of the data set
    :return: the values of the samples, the skin type fo the samples, the ids of the genes and the ids of the samples
    """

    # Read and extract all viable data
    series_data, sample_data, sample_matrix = read_data(name)
    # series = extract_series_data(series_data)
    samples = extract_sample_data(sample_data)
    sample_ids, gene_ids, sample_values = extract_sample_matrix(sample_matrix)

    skin_types = []
    disease_type = 'unknown'

    # Every sample has a different way of testing which skin type
    for sample in samples:
        if name == 'GSE36842':
            skin_types.append(sample['title'].split(' ')[-1])
            disease_type = 'AD'
        elif name == 'GSE32924':
            skin_types.append(sample['title'].split(' ')[0])
            disease_type = 'AD'
        elif name == 'GSE27887':
            skin_types.append(' '.join(sample['title'].split(' ')[1:3]))
            disease_type = 'AD'
        elif name == 'GSE14905':
            skin_types.append(sample['title'].split(' ')[-1].split('-')[0])
            disease_type = 'psoriasis'
        elif name == 'GSE78097':
            skin_types.append(' '.join(sample['title'].split(' ')[2:-1]))
            disease_type = 'psoriasis'
        elif name == 'GSE41662':
            skin_types.append(sample['title'].split(' ')[-1])
            disease_type = 'psoriasis'
        elif name == 'GSE34248':
            skin_types.append(sample['title'].split('-')[-1])
            disease_type = 'psoriasis'
        elif name == 'GSE30999':
            skin_types.append(sample['title'].split('_')[-1])
            disease_type = 'psoriasis'
        elif name == 'GSE13355':
            skin_types.append(sample['title'].split('_')[-2])
            disease_type = 'psoriasis'
        else:
            raise NotImplementedError('No extraction made for this data type')

    # Hot encode the data
    skin_types = encode_output(np.asarray(skin_types), disease_type)

    return sample_values, skin_types, gene_ids, sample_ids