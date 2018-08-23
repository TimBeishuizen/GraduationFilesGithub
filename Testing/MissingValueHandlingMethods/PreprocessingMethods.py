from sklearn import model_selection as SMS, linear_model as SLM, metrics as SME, preprocessing as PP

import numpy as np


def hot_encode_categorical_values(X, hot_encoders=None, missing_locations=None, missing_values=None):
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
        new_col = np.copy(X[:, i:i+1])

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)

        except:

            if hot_encoders is None:
                # Create hot encoder en use it for fitting and transformation
                hot_encoder = PP.MultiLabelBinarizer()
                new_col = hot_encoder.fit_transform(new_col)
            else:
                new_col = hot_encoders[i].transform(new_col)

        # Keep record of the new data and
        new_X = np.append(new_X, new_col, axis=1)

    return new_X


def scale_numerical_values(X, scalers=None, missing_locations=None, missing_values=None):
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
                scaler = PP.StandardScaler()
                new_col = scaler.fit_transform(new_col)
            else:
                new_col = scalers[i].transform(new_col)

        except:

            new_col = new_col

        # Keep record of the new data and
        new_X = np.append(new_X, new_col, axis=1)

    return new_X


def find_hot_encoders(X, missing_values=None):
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
            hot_encoder = PP.MultiLabelBinarizer()
            new_col = hot_encoder.fit_transform(new_col)

        # Keep record of the new data and
        hot_encoders.append(hot_encoder)

    return hot_encoders


def find_scalers(X, missing_values=None):
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
            scaler = PP.StandardScaler()
            scaler.fit(new_col)
        except:

            # Create hot encoder en use it for fitting and transformation
            scaler = None

        # Keep record of the new data and
        scalers.append(scaler)

    return scalers

def compute_scores(X, y):
    """

    :param X: 
    :param y: 
    :return: 
    """

    # Split data in training and test data.
    X_train, X_test, y_train, y_test = SMS.train_test_split(X, y)

    # Cross validation data
    cross_val = SMS.LeaveOneOut()
    # cross_val = SMS.KFold(n_splits=10)
    val_score = []

    print("Starting cross validation")
    # Compute validation score
    for train_index, test_index in cross_val.split(X_train):
        ml = SLM.LogisticRegression()
        ml.fit(X_train[train_index], y_train[train_index])
        val_score.append(ml.score(X_train[test_index], y_train[test_index]))

    # Compute the mean of the validation score
    # T_val.append(np.mean(np.asarray(val_score)))

    # Compute test score
    ml = SLM.LogisticRegression()
    ml.fit(X_train, y_train)
    # T_test.append(ml.score(X_test, y_test))
    y_pred = ml.predict(X_test)
    prec, rec, Fbeta, _ = SME.precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # precision.append(prec)
    # recall.append(rec)
    # F1.append(Fbeta)

    return np.mean(np.asarray(val_score)), ml.score(X_test, y_test), prec, rec, Fbeta


