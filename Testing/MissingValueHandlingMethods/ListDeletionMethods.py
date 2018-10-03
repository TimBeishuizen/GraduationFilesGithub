import numpy as np

from sklearn import preprocessing as PP

def cca(X, y, missing_values=None):
    """

    :param X:
    :param y:
    :return:
    """

    # Make np arrays of the input
    X = np.asarray(X)
    y = np.asarray(y)

    # find missing locations
    missing_loc = np.argwhere(X == missing_values)
    missing_rows = np.unique(missing_loc[:, 0])

    # Delete the rows with missing values
    new_X = np.delete(X, missing_rows, axis=0)
    new_y = np.delete(y, missing_rows, axis=0)

    return new_X, new_y


def aca(X, features, missing_values=None, important_features=None, removal_fraction=None):
    """

    :param X:
    :param y:
    :param missing_values:
    :param important_features:
    :param removal_fraction:
    :return:
    """

    # Define removal_percentage if not defined:
    if removal_fraction is None:
        removal_fraction = 0

    if important_features is None:
        important_features = []

    # Make np arrays of the input
    X = np.asarray(X)

    # find missing locations
    missing_loc = np.argwhere(X == missing_values)
    missing_columns = missing_loc[:, 1]
    unique_columns = np.sort(np.unique(missing_columns))
    count_columns = [np.count_nonzero(missing_columns == column) for column in unique_columns]

    new_X = np.copy(X)
    new_features = np.copy(features)

    # Remove features if not deemed important and if a big enough fraction is missing
    for i in reversed(range(unique_columns.shape[0])):
        if unique_columns[i] not in important_features and removal_fraction < count_columns[i] / X.shape[0]:
            new_X = np.delete(new_X, i, axis=1)
            new_features = np.delete(new_features, i, axis=0)

    # Return deleted data, after removing last missing values with CCA
    return new_X, new_features


def wca(X, missing_values=None):
    """

    :param X:
    :param y:
    :param missing_values:
    :return:
    """

    # Make np arrays of the input
    X = np.asarray(X)
    new_X = np.copy(X)

    # Find missing locations
    missing_loc = np.argwhere(X == missing_values)
    missing_rows = np.unique(missing_loc[:, 0])

    # Function to convert booleans to 0 and 1
    def convert_bool(dataset):
        """

        :param dataset:
        :return:
        """

        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                if dataset[i, j] == 'False':
                    dataset[i, j] = 0
                elif dataset[i, j] == 'True':
                    dataset[i, j] = 1

        return dataset

    # Find complete rows and scale data
    complete_rows = np.delete(np.asarray(range(X.shape[0])), missing_rows, axis=0)
    scaler = PP.StandardScaler()
    complete_data = X[complete_rows, :]

    scaled_data = scaler.fit_transform(convert_bool(np.copy(complete_data)).astype(float))

    # Function to find nearest neighbours
    def nearest_neighbour(node, nodes):
        """

        :param node:
        :param nodes:
        :return:
        """
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return np.argmin(dist_2)

    # Initialization of nearest neighbours
    neighbours = []

    for row in missing_rows:
        missing_row = np.asarray([X[row, :]])

        # Find missing locations and store non-missing locations
        missing_locs = missing_loc[np.argwhere(missing_loc[:, 0] == row), 1]
        complete_locs = np.delete(np.asarray(range(X.shape[1])), missing_locs, axis=0)

        # Dummies for missing values and scale values
        for loc in missing_locs:
            missing_row[:, loc] = 0
        scaled_row = scaler.transform(convert_bool(missing_row).astype(float))

        # Remove values from scaled row and complete data
        scaled_row = scaled_row[:, complete_locs]
        temp_scaled_data = scaled_data[:, complete_locs]

        new_X[row, :] = complete_data[nearest_neighbour(scaled_row, temp_scaled_data), :]

    return new_X
