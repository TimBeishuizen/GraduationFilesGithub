import numpy as np
import scipy.stats as stats

from sklearn import neighbors
from sklearn import linear_model as LM

from MissingValueHandlingMethods import PreprocessingMethods as PM


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

    # Find hot encoders for categorical data
    hot_encoders = PM.find_hot_encoders(X, missing_values=missing_values)

    # Find scalers for numerical data
    scalers = PM.find_scalers(X, missing_values=missing_values)

    # Find imputation value for every missing value
    for i in range(missing_loc.shape[0]):

        # Find missing row and missing var in example:
        missing_row = X[missing_loc[i, 0]:missing_loc[i, 0] + 1, :]
        output_var = missing_loc[i, 1]

        # Find all missing vars in example, also that are not output:
        missing_vars = np.argwhere(missing_row == missing_values)[:, 1]
        extra_missing_vars = np.delete(missing_vars, np.argwhere(missing_vars == missing_loc[i, 1]))

        # Find remaining variables for rest of the data set
        if extra_missing_vars.tolist() == []:
            remaining_vars = np.asarray(range(X.shape[1]))
        else:
            remaining_vars = np.delete(np.asarray(range(X.shape[1])), extra_missing_vars.tolist(), axis=0)

        # Remove additional missing variables in the missing row and select new location for missing value
        reduced_X = np.copy(X[:, remaining_vars])

        # Find missing locations in rest of dataset
        reduced_missing_loc = np.argwhere(reduced_X == missing_values)
        reduced_missing_rows = np.unique(reduced_missing_loc[:, 0])

        # Find complete rows and scale data
        reduced_complete_rows = np.delete(np.asarray(range(reduced_X.shape[0])), reduced_missing_rows, axis=0)
        complete_data = X[reduced_complete_rows, :]

        try:
            complete_data[:, output_var].astype(float)

            kNN = neighbors.KNeighborsRegressor(n_neighbors=k)

            output = complete_data[:, output_var].astype(float)

        except ValueError:
            kNN = neighbors.KNeighborsClassifier(n_neighbors=k)

            output = complete_data[:, output_var]

        # Remove missing rows, add scaling and hot encoding
        scaled_data = PM.scale_numerical_values(complete_data, scalers, missing_vars, missing_values)
        processed_data = PM.hot_encode_categorical_values(scaled_data, hot_encoders, missing_vars, missing_values)

        # Select the sample with the missing value, scale it and remove the missing value
        scaled_missing_row = PM.scale_numerical_values(missing_row, scalers, output_var, missing_values)
        processed_missing_row = PM.hot_encode_categorical_values(scaled_missing_row, hot_encoders, output_var, missing_values)

        kNN.fit(processed_data, output)

        new_X[missing_loc[i, 0], missing_loc[i, 1]] = kNN.predict(processed_missing_row)[0]

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

    # Find hot encoders for categorical data
    hot_encoders = PM.find_hot_encoders(X, missing_values=missing_values)

    # Find scalers for numerical data
    scalers = PM.find_scalers(X, missing_values=missing_values)

    # Find imputation value for every missing value
    for i in range(missing_loc.shape[0]):

        # Find missing row and missing var in example:
        missing_row = X[missing_loc[i, 0]:missing_loc[i, 0] + 1, :]
        output_var = missing_loc[i, 1]

        # Find all missing vars in example, also that are not output:
        missing_vars = np.argwhere(missing_row == missing_values)[:, 1]
        extra_missing_vars = np.delete(missing_vars, np.argwhere(missing_vars == missing_loc[i, 1]))

        # Find remaining variables for rest of the data set
        if extra_missing_vars.tolist() == []:
            remaining_vars = np.asarray(range(X.shape[1]))
        else:
            remaining_vars = np.delete(np.asarray(range(X.shape[1])), extra_missing_vars.tolist(), axis=0)

        # Remove additional missing variables in the missing row and select new location for missing value
        reduced_X = np.copy(X[:, remaining_vars])

        # Find missing locations in rest of dataset
        reduced_missing_loc = np.argwhere(reduced_X == missing_values)
        reduced_missing_rows = np.unique(reduced_missing_loc[:, 0])

        # Find complete rows and scale data
        reduced_complete_rows = np.delete(np.asarray(range(reduced_X.shape[0])), reduced_missing_rows, axis=0)
        complete_data = X[reduced_complete_rows, :]

        try:
            complete_data[:, output_var].astype(float)

            linear_model = LM.LinearRegression()

            output = complete_data[:, output_var].astype(float)

        except ValueError:
            linear_model = LM.LogisticRegression()

            output = complete_data[:, output_var]

        # Remove missing rows, add scaling and hot encoding
        scaled_data = PM.scale_numerical_values(complete_data, scalers, missing_vars, missing_values)
        processed_data = PM.hot_encode_categorical_values(scaled_data, hot_encoders, missing_vars, missing_values)

        # Select the sample with the missing value, scale it and remove the missing value
        scaled_missing_row = PM.scale_numerical_values(missing_row, scalers, output_var, missing_values)
        processed_missing_row = PM.hot_encode_categorical_values(scaled_missing_row, hot_encoders, output_var, missing_values)

        linear_model.fit(processed_data, output)

        new_X[missing_loc[i, 0], missing_loc[i, 1]] = linear_model.predict(processed_missing_row)[0]

    return new_X