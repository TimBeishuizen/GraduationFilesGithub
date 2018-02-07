import os
import random

# Location of the gen dataset
# http://www.fruitfly.org/sequence/human-datasets.html
gen_data_loc = r'C:\Users\s119104\Documents\Studie\GraduationProject\DataSets\Splice set'


def read_gene_file(data_type, result_type, gene_type):
    """ Reads the gen file
    
    :param data_type: Defines the type for machine learning: train or test
    :param result_type: Defines the type of output value: True or False
    :param gene_type: Defines the type of gene: acceptor or donor
    :return: a file with the correct genetic data
    """

    # First go to the right location
    os.chdir(gen_data_loc)

    file_name = 'splice.'

    if data_type in {'train', 'test'}:
        file_name = file_name + data_type
    else:
        raise ValueError("Not the right name for the data type")

    if type(result_type) != bool:
        raise ValueError("Result_type is not a boolean")
    elif result_type:
        file_name = file_name + '-real'
    else:
        file_name = file_name + '-false'

    if gene_type == 'acceptor':
        file_name = file_name + '.A'
    elif gene_type == 'donor':
        file_name = file_name + '.D'

    file = open(file_name, 'r')

    return file.read()







