import numpy as np

from sklearn import svm as SVM
from sklearn import ensemble as SEN


def embedded_methods_classifier():
    return None


def embedded_forward_selection(X, y, features, ML_alg='svm', filter_method='Rank', threshold=0.05):
    """ Use embedded forward selection to select a subset of features

    :param X: A numpy matrix witth the values for the features for every sample
    :param y: The classes of every sample
    :param features: A list with the names of the features
    :param ML_alg: The machine learning algorithm that is used for evaluation. Algorithms: support vector machines
                ('svm', default), random forests ('rf')
    :param filter_method: The way to reduce the number of features: 'Rank' (default), 'Population'
    :param threshold: The threshold for the filter method
    :return: The reduced number of features and their values
    """

    y = np.asarray(y)
    features = np.asarray(features)

    # Making the arrays robust
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have a different number of samples")
    elif X.shape[1] != features.shape[0]:
        raise ValueError("There is a different number of features in X and in the name vector")

    # Making the method values robust
    if ML_alg not in ["svm", "rf", "gb"]:
        raise ValueError("Unknown ranking method")
    elif filter_method not in ["Rank", "Population"]:
        raise ValueError("Unknown filter method")
    elif type(threshold) not in [int, float]:
        raise ValueError("Not a suitable threshold value")
    elif filter_method == "Population" and type(threshold) == float and (threshold < 0 or threshold > 1):
        raise ValueError("Population error should be integer (flat number) or float between 0 and 1 (fraction)")

    # Make population threshold an integer
    if filter_method == "Population" and 0 < threshold < 1:
        threshold = int(threshold * X.shape[1])

    # Find ranks and order
    ordered_X, ordered_features, ranks = order_ranks(X, y, features, ML_alg)

    if filter_method == 'Rank':
        n_features = np.count_nonzero(ranks > threshold)
        X_new = ordered_X[:, :n_features]
        features_new = ordered_features[:n_features]
    elif filter_method == 'Population' and type(threshold) == int:
        X_new = ordered_X[:, :threshold]
        features_new = ordered_features[:threshold]
    else:
        raise ValueError

    if filter_method == "Rank":
        alternative_threshold = len(features_new)
    elif filter_method == "Population":
        alternative_threshold = ranks[threshold + 1]
    else:
        raise ValueError

    # return new X and features values
    return X_new, features_new, alternative_threshold


def embedded_backward_elimination(X, y, features, ML_alg='svm', filter_method='Rank', threshold=0.05, n_elimination=1):
    """ Use embedded forward selection to select a subset of features

    :param X: A numpy matrix witth the values for the features for every sample
    :param y: The classes of every sample
    :param features: A list with the names of the features
    :param ML_alg: The machine learning algorithm that is used for evaluation. Algorithms: support vector machines
                ('svm', default), random forests ('rf')
    :param filter_method: The way to reduce the number of features: 'Rank' (default), 'Population'
    :param threshold: The threshold for the filter method
    :return: The reduced number of features and their values
    """

    y = np.asarray(y)
    features = np.asarray(features)

    # Making the arrays robust
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have a different number of samples")
    elif X.shape[1] != features.shape[0]:
        raise ValueError("There is a different number of features in X and in the name vector")

    # Making the method values robust
    if ML_alg not in ["svm", "rf", "gb"]:
        raise ValueError("Unknown ranking method")
    elif filter_method not in ["Rank", "Population"]:
        raise ValueError("Unknown filter method")
    elif type(threshold) not in [int, float]:
        raise ValueError("Not a suitable threshold value")
    elif filter_method == "Population" and type(threshold) == float and (threshold < 0 or threshold > 1):
        raise ValueError("Population error should be integer (flat number) or float between 0 and 1 (fraction)")

    # Make population threshold an integer
    if filter_method == "Population" and 0 < threshold < 1:
        threshold = int(threshold * X.shape[1])

    # Initialize candidate
    X_new = X
    feat_new = features
    feat_candidate = []
    rank = None

    while set(feat_candidate) != set(feat_new):
        # Define candidate
        feat_candidate = feat_new

        # Order the features
        X_new, feat_new, ranks = order_ranks(X_new, y, feat_new, ML_alg)

        # Filter worst feature out if needed
        for i in range(n_elimination):
            if filter_method == "Rank" and ranks[-1-i] < threshold:
                X_new = X_new[:, :-1]
                feat_new = feat_new[:-1]
            elif filter_method == "Population" and len(feat_new) > threshold:
                X_new = X_new[:, :-1]
                feat_new = feat_new[:-1]
            # Unnecessary, but speeds up the last time
            else:
                break


        rank = ranks[-1]
        print("Iterating... %i features, lowest rank %f" %(len(feat_new), rank))

    if filter_method == "Rank":
        final_threshold = len(feat_new)
    elif filter_method == "Population":
        final_threshold = rank
    else:
        raise ValueError


    return X_new, feat_new, final_threshold





def order_ranks(X, y, features, ML_alg="svm"):
    """ Orders the matrices by rank

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The output of every sample
    :param ML_alg: The method the features will be ranking with: Algorithms: support vector machines
                ('svm', default), random forests ('rf')
    :param ranking_settings: The extra settings for the ranking method: None (default)
    :return: Ordered ranks
    """

    # Give all methods their ranking
    if ML_alg == 'svm':
        # Create and fit svc
        svm = SVM.LinearSVC()
        svm.fit(X, y)

        # Find the weights
        weights = svm.coef_
        ranks = np.mean(np.abs(weights), axis=0)

        # Order the weights
        order = np.flip(np.argsort(ranks), axis=0)
    elif ML_alg == 'rf':
        # Create and fit random forest
        rf = SEN.RandomForestClassifier(n_estimators=10000)
        rf.fit(X, y)

        # Find the weights
        ranks = rf.feature_importances_
        order = np.flip(np.argsort(ranks), axis=0)
    elif ML_alg == 'gb':
        # Create and fit random forest
        gb = SEN.GradientBoostingClassifier()
        gb.fit(X, y)

        # Find the weights
        ranks = gb.feature_importances_
        order = np.flip(np.argsort(ranks), axis=0)
    else:
        raise ValueError

    ordered_X = np.squeeze(X[:, order])
    ordered_features = np.squeeze(features[order])
    ordered_ranks = ranks[order]

    return ordered_X, ordered_features, ordered_ranks
