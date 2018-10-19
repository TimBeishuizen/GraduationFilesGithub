import csv
import numpy as np


def read_csv_dataset(csv_path):
    """ Reads in the dataset from a csv-file. This file needs to be based on the following description:
    - Rows correspond to instances and columns correspond to features
    - The first entry every column corresponds to the name of the feature with all other entries corresponding to
    values of that feature
    - The last entry of every row corresponds to the output labels of every instance.

    :param csv_path: The location and name of the csv-file
    :return: The data in a matrix, the output labels and feature names
    """

    matrix = []

    # Opening CSV file
    with open(csv_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            matrix.append(row)

    # Extracting X, y and feature values
    features = np.asarray(matrix)[0, :-1]

    X = np.asarray(matrix)[1:, :-1]

    # Make output numeric if possible
    try:
        y = np.asarray(matrix)[1:, -1].astype(float)
    except ValueError:
        y = np.asarray(matrix)[1:, -1]

    return X, y, features