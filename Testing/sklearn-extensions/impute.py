# -*- coding: utf-8 -*-
"""Method for missing value imputation"""

# Authors: T.P.A. Beishuizen <tim.beishuizen@gmail.com>

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

from sklearn.linear_model import LinearRegression, LogisticRegression

import numpy as np


def LearningImputation(X, classification_estimator=LinearRegression(), regression_estimator=LogisticRegression(),
                       missing_values=None):
    """

    :param X:
    :param classification_estimator:
    :param regression_estimator:
    :param missing_values:
    :return:
    """

    # Make np arrays of the input
    X = np.copy(np.asarray(X))
    new_X = np.copy(X)

    # Find missing locations
    missing_loc = np.argwhere(X == missing_values)
    loc_missing_var = [[]] * X.shape[1]

    for i in range(missing_loc.shape[0]):
            loc_missing_var[missing_loc[i, 1]].append(missing_loc[i, 0])

    # Find hot encoders for categorical data
    hot_encoders = _find_temp_hot_encoders(X, missing_values=missing_values)

    # Find scalers for numerical data
    scalers = _find_temp_scalers(X, missing_values=missing_values)

    # Find imputation value for every missing value
    for i in range(missing_loc.shape[0]):

        if (i + 1) % 500 == 0:
            print("Currently at %i of %i" %(i, missing_loc.shape[0]))

        # Find missing row and missing var in example:
        missing_row = X[missing_loc[i, 0]:missing_loc[i, 0] + 1, :]
        output_var = missing_loc[i, 1]

        # Find all missing vars in example, also that are not output:
        missing_vars = np.argwhere(missing_row == missing_values)[:, 1]
        extra_missing_vars = np.delete(missing_vars, np.argwhere(missing_vars == missing_loc[i, 1]))

        missing_samples = [missing_loc[i, 0]]

        for var in range(X.shape[1]):
            if var not in extra_missing_vars:
                missing_samples.extend(loc_missing_var[var])

        removing_samples = np.unique(missing_samples)

        complete_data = np.delete(np.copy(X), removing_samples, 0)

        try:
            complete_data[:, output_var].astype(float)

            linear_model = clone(classification_estimator)

            output = complete_data[:, output_var].astype(float)

        except ValueError:
            linear_model = clone(regression_estimator)

            if len(np.unique(complete_data[:, output_var]).tolist()) == 1:
                new_X[missing_loc[i, 0], missing_loc[i, 1]] = np.unique(complete_data[:, output_var])[0]
                continue

            output = complete_data[:, output_var]

        # Remove missing rows, add scaling and hot encoding
        scaled_data = _temp_scale_numeric_values(complete_data, scalers, missing_vars, missing_values)
        processed_data = _temp_hot_encode_categorical_values(scaled_data, hot_encoders, missing_vars, missing_values)

        # Select the sample with the missing value, scale it and remove the missing value
        scaled_missing_row = _temp_scale_numeric_values(missing_row, scalers, missing_vars, missing_values)
        processed_missing_row = _temp_hot_encode_categorical_values(scaled_missing_row, hot_encoders, missing_vars, missing_values)

        linear_model.fit(processed_data, output)

        new_X[missing_loc[i, 0], missing_loc[i, 1]] = linear_model.predict(processed_missing_row)[0]

    return new_X


def _find_temp_hot_encoders(X, missing_values=None):
    """

    :param X:
    :param missing_values:
    :return:
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))
    hot_encoders = []

    for i in range(X.shape[1]):

        # Copy a new row and delete the missing values
        new_col = np.copy(X[:, i:i + 1])
        new_col = np.delete(new_col, new_col == missing_values, axis=0)

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)
            hot_encoder = None
        except:

            # Create hot encoder en use it for fitting and transformation
            hot_encoder = MultiLabelBinarizer()
            new_col = hot_encoder.fit_transform(new_col)

        # Keep record of the new data and
        hot_encoders.append(hot_encoder)

    return hot_encoders


def _find_temp_scalers(X, missing_values=None):
    """

    :param X:
    :param missing_values:
    :return:
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))
    scalers = []

    for i in range(X.shape[1]):

        # Copy a new row and delete the missing values
        new_col = np.copy(X[:, i:i + 1])
        new_col = np.delete(new_col, new_col == missing_values, axis=0)

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)
            scaler = MinMaxScaler()
            scaler.fit(new_col)
        except:

            # Create hot encoder en use it for fitting and transformation
            scaler = None

        # Keep record of the new data and
        scalers.append(scaler)

    return scalers


def _temp_hot_encode_categorical_values(X, hot_encoders=None, missing_locations=None, missing_values=None):
    """

    :param X:
    :param hot_encoders:
    :return:
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))

    for i in range(X.shape[1]):

        if np.any(missing_locations == i) or np.any(X[:, i] == missing_values):
            continue

        # Copy a new row
        new_col = np.copy(X[:, i:i + 1])

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)

        except:
            if hot_encoders is None:
                # Create hot encoder en use it for fitting and transformation
                hot_encoder = MultiLabelBinarizer()
                new_col = hot_encoder.fit_transform(new_col)
            else:
                new_col = hot_encoders[i].transform(new_col)

        # Keep record of the new data and
        new_X = np.append(new_X, new_col, axis=1)

    return new_X


def _temp_scale_numeric_values(X, scalers=None, missing_locations=None, missing_values=None):
    """

    :param X:
    :param scalers:
    :return:
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))

    for i in range(X.shape[1]):

        # Copy a new row and delete the missing values
        new_col = np.copy(X[:, i:i + 1])
        # new_col = np.delete(new_col, new_col == missing_values, axis=0)

        if np.any(missing_locations == i) or np.any(X[:, i] == missing_values):
            new_X = np.append(new_X, new_col, axis=1)
            continue

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)

            if scalers is None:
                # Create hot encoder en use it for fitting and transformation
                scaler = MinMaxScaler()
                new_col = scaler.fit_transform(new_col)
            else:
                new_col = scalers[i].transform(new_col)

        except:

            new_col = new_col

        # Keep record of the new data and
        new_X = np.append(new_X, new_col, axis=1)

    return new_X
