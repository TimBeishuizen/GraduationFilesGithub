from sklearn import preprocessing as PP

from sklearn import model_selection as SMS, linear_model as SLM, metrics as SME

import numpy as np

def hot_encode_categorical_values(X):
    """
    
    :param X: 
    :return: 
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))

    for i in range(X.shape[1]):
        try:
            new_col = np.copy(X[:, i:i+1].astype(float))
        except:
            new_col = PP.MultiLabelBinarizer().fit_transform(np.copy(X)[:, i])

        new_X = np.append(new_X, new_col, axis=1)

    return new_X


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


