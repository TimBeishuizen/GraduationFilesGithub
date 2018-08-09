import numpy as np
import scipy.stats as stats

from sklearn import neighbors
from sklearn import preprocessing as PP
from sklearn import linear_model as LM

def value_imputation(X, missing_values=None, imputation_value=0):
    """

    :param X:
    :param y:
    :param missing_values:
    :param imputation_value:
    :return:
    """

    if not isinstance(imputation_value, list):
        imputation_value = [imputation_value] * X.shape[1]

    X = np.copy(np.asarray(X))

    # Find locations of the missing values
    missing_loc = np.argwhere(X == missing_values)
    new_X = np.copy(X)

    # Add values to missing values
    for i in range(missing_loc.shape[0]):
        new_X[missing_loc[i, 0], missing_loc[i, 1]] = imputation_value[missing_loc[0, 1]]

    return new_X

def mean_imputation(X, missing_values=None, imputation_type='mean'):
    """

    :param X:
    :param y:
    :param imputation_type:
    :return:
    """

    X = np.copy(np.asarray(X))

    # Find of imputation type was given correctly
    if imputation_type in ['mean', 'median', 'mode']:
        imputation_type = [imputation_type] * X.shape[1]
    elif isinstance(imputation_type, list):
        for single_imputation in imputation_type:
            if single_imputation not in ['mean', 'median', 'mode']:
                raise ValueError('One of the imputation types is not within the implemented types: '
                                 'mean, median and mode')
    else:
        raise ValueError('The imputation type is not within the implemented types: mean, median and mode')

    imputation_values = []

    # First find the values to imputed
    for j in range(X.shape[1]):

        # Foolproof if categorical values, while not given categorical values
        complete_values = np.delete(X[:, j], np.argwhere(X[:, j] == missing_values))
        unique_values = np.unique(complete_values)
        try:
            unique_values.astype(float)
            if unique_values.shape[0] <= 2:
                imputation_type[j] = 'median'
        except ValueError:
            imputation_type[j] = 'mode'

        if imputation_type[j] == 'mean':
            imputation_values.append(np.mean(complete_values.astype(float)))

        elif imputation_type[j] == 'median':
            imputation_values.append(np.median(complete_values.astype(float)))

        elif imputation_type[j] == 'mode':
            missing_removed = np.argwhere(X[:, j] != missing_values)
            new_mode, _ = stats.mode(X[missing_removed, j], nan_policy='omit')
            imputation_values.append(np.asscalar(new_mode))

    # Find locations of the missing values
    missing_loc = np.argwhere(X == missing_values)
    new_X = np.copy(X)

    # Add values to missing values
    for i in range(missing_loc.shape[0]):
        new_X[missing_loc[i, 0], missing_loc[i, 1]] = imputation_values[missing_loc[i, 1]]

    return new_X


def hot_deck_imputation(X, missing_values=None):
    """

    :param X:
    :param y:
    :param missing_values:
    :return:
    """

    X = np.copy(np.asarray(X))

    # Find locations of the missing values
    missing_loc = np.argwhere(X == missing_values)
    new_X = np.copy(X)

    # Add values to missing values
    for i in range(missing_loc.shape[0]):
        missing_removed = np.argwhere(X[:, missing_loc[i, 1]] != missing_values)[:, 0]
        new_val = np.random.choice(X[missing_removed, missing_loc[i, 1]], 1)
        new_X[missing_loc[i, 0], missing_loc[i, 1]] = np.asscalar(new_val)

    return new_X


def missing_indicator_imputation(X, missing_values=None):
    """

    :param X:
    :param y:
    :param missing_values:
    :param imputation_value:
    :return:
    """

    X = np.copy(np.asarray(X))

    # Find locations of the missing values
    missing_loc = np.argwhere(X == missing_values)
    missing_columns = missing_loc[:, 1]
    unique_columns = np.sort(np.unique(missing_columns))
    new_X = np.copy(X)

    # Add values to missing values
    for i in unique_columns:
        mi_row = np.asarray(X[:, i] != missing_values).astype(int).reshape([-1, 1])
        new_X = np.append(new_X, mi_row, axis=1)

    return new_X


def kNN_imputation(X, missing_values=None, k=1):
    """

    :param X:
    :param y:
    :param missing_values:
    :param k:
    :return:
    """

    # Make np arrays of the input
    X = np.copy(np.asarray(X))

    new_X = np.copy(X)

    # Find missing locations
    missing_loc = np.argwhere(X == missing_values)
    missing_rows = np.unique(missing_loc[:, 0])

    # Find complete rows and scale data
    complete_rows = np.delete(np.asarray(range(X.shape[0])), missing_rows, axis=0)
    complete_data = X[complete_rows, :]

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
                elif dataset[i, j] == missing_values: # Only made for removal of missing values in nearest neighbour
                    dataset[i, j] = 0.5

        return dataset

    # Scale the complete data
    scaler = PP.StandardScaler()
    scaled_data = scaler.fit_transform(convert_bool(np.copy(complete_data)).astype(float))

    # Find imputation value for every missing value
    for i in range(missing_loc.shape[0]):

        try:
            complete_data[:, missing_loc[i, 1]].astype(float)

            kNN = neighbors.KNeighborsRegressor(n_neighbors=k)

            output = complete_data[:, missing_loc[i, 1]].astype(float)

        except ValueError:
            kNN = neighbors.KNeighborsClassifier(n_neighbors=k)

            output = complete_data[:, missing_loc[i, 1]]

        input = np.delete(np.copy(scaled_data), missing_loc[i, 1], axis=1)

        # Select the sample with the missing value, scale it and remove the missing value
        missing_row = X[missing_loc[i, 0]:missing_loc[i, 0] + 1, :]
        scaled_missing_row = scaler.transform(convert_bool(missing_row).astype(float))
        missing_input = np.delete(scaled_missing_row, missing_loc[i, 1], axis=1)

        kNN.fit(input.astype(float), output)

        new_X[missing_loc[i, 0], missing_loc[i, 1]] = kNN.predict(missing_input)[0]

    return new_X


def regression_imputation(X, missing_values=None):
    """

    :param X:
    :param y:
    :param missing_values:
    :return:
    """

    # Make np arrays of the input
    X = np.copy(np.asarray(X))

    new_X = np.copy(X)

    # Find missing locations
    missing_loc = np.argwhere(X == missing_values)
    missing_rows = np.unique(missing_loc[:, 0])

    # Find complete rows and scale data
    complete_rows = np.delete(np.asarray(range(X.shape[0])), missing_rows, axis=0)
    complete_data = X[complete_rows, :]

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
                elif dataset[i, j] == missing_values: # Only made for removal of multiple missing values in nearest neighbour
                    dataset[i, j] = 0.5

        return dataset

    # Scale the complete data
    scaler = PP.StandardScaler()
    scaled_data = scaler.fit_transform(convert_bool(np.copy(complete_data)).astype(float))

    # Find imputation value for every missing value
    for i in range(missing_loc.shape[0]):

        try:
            complete_data[:, missing_loc[i, 1]].astype(float)

            linear_model = LM.LinearRegression()

            output = complete_data[:, missing_loc[i, 1]].astype(float)

        except ValueError:
            linear_model = LM.LogisticRegression()

            output = complete_data[:, missing_loc[i, 1]]

        input = np.delete(np.copy(scaled_data), missing_loc[i, 1], axis=1)

        # If only one value is left
        if np.unique(output).shape[0] == 1:
            new_X[missing_loc[i, 0], missing_loc[i, 1]] = np.unique(output)[0]
            continue

        # Select the sample with the missing value, scale it and remove the missing value
        missing_row = X[missing_loc[i, 0]:missing_loc[i, 0] + 1, :]
        scaled_missing_row = scaler.transform(convert_bool(missing_row).astype(float))
        missing_input = np.delete(scaled_missing_row, missing_loc[i, 1], axis=1)

        linear_model.fit(input.astype(float), output)

        new_X[missing_loc[i, 0], missing_loc[i, 1]] = linear_model.predict(missing_input)[0]

    return new_X