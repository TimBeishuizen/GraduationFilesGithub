from DataExtraction import SkinDataExtraction as SDE
import numpy as np
import csv

def import_example_data(data_name):
    """ Import one of the example data sets

    :param data_name: The name of the data set. Currently it is possible to choose from:
        - 'Psoriasis' (Microarray),
        - 'RSCTC' (Microarray),
        - 'Arcene' (Mass Spectometry)
        - 'MicroOrganisms' (Mass Spectometry)
    :return: The sample values (X), the class values (y) and the variable values (features)
    """



    if data_name == 'Psoriasis':
        print("Importing %s dataset..." % data_name)
        psoriasis_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905']
        return SDE.extract_multiple_data_sets(psoriasis_names, normalization=True)
    elif data_name == 'RSCTC':
        table_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\RSCTCMicroArraySet\RSCTC_micro_array_data.csv'
    elif data_name == 'Arcene':
        table_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\ArceneMassSpectometrySet\arcene_mass_spect_data.csv'
    elif data_name == 'MicroOrganisms':
        table_path = r'C:\Users\s119104\Documents\GitHub\GraduationFilesGithub\Testing\DataSets\BioMassSpectometrySet\bio_mass_spect_data.csv'
    else:
        raise ValueError('A data set with reference name "%s" does not exist' % data_name)

    print("Importing %s dataset..." % data_name)

    matrix = []

    # Opening CSV file
    with open(table_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            matrix.append(row)

    # Extracting X, y and feature values
    features = np.asarray(matrix)[0, :-1]
    X = np.asarray(matrix)[1:, :-1].astype(float)
    y = np.asarray(matrix)[1:, -1].astype(int)

    return X, y, features
