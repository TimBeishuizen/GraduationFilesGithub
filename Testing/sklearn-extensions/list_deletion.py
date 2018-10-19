# -*- coding: utf-8 -*-
"""Generic list deletion mixin"""

# Author: T.P.A. Beishuizen <tim.beishuizen@gmail.com>

import numpy as np

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