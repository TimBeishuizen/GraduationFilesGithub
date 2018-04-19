import numpy as np
import scipy.stats as sp

import sklearn.feature_selection as FS


def filter_methods_classifier(X, y, features, ranking_method='T-test', filter_method='Rank', threshold=0.05,
                              ranking_settings='Default'):
    """ 
    
    :param X: A numpy matrix with the values for the features for every sample
    :param y: The classes of every sample
    :param features: A list with the names of the features
    :param ranking_method: The method the features will be ranking with: "T-test" (default), "Mutual information"
    :param filter_method: The method that is used to filter the ranked features: "Rank" (default), "Population"
    :param threshold: The value for the filter method to be used. If filter_method is "Population value", values in [0, 1]
                        are used to give fraction of the population. Integers give flat number of features to be chosen
    :param ranking_settings: The extra settings for the ranking method: "Default" (default), test_type in t-test,
                                MI_type in mutual information
    :return: The matrix with reduced features, as well as the names of the features that are kept
    """

    y = np.asarray(y)
    features = np.asarray(features)

    # Making the arrays robust
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have a different number of samples")
    elif X.shape[1] != features.shape[0]:
        raise ValueError("There is a different number of features in X and in the name vector")

    # Making the method values robust
    if ranking_method not in ["T-test", "Mutual information", "MI"]:
        raise ValueError("Unknown ranking method")
    elif filter_method not in ["Rank", "Population"]:
        raise ValueError("Unknown filter method")
    elif type(threshold) not in [int, float]:
        raise ValueError("Not a suitable threshold value")
    elif filter_method == "Population" and type(threshold) == float and (threshold < 0 or threshold > 1):
        raise ValueError("Population error should be integer (flat number) or float between 0 and 1 (fraction)")

    X, features, ranks = order_ranks(X, y, features, ranking_method=ranking_method, ranking_settings=ranking_settings)

    # Use filter method to remove ranks
    if filter_method == 'Rank' and ranking_method == "T-test":
        n_features = np.count_nonzero(ranks <= threshold)
        X_new = X[:, :n_features]
        features_new = features[:n_features]
    elif filter_method == 'Rank' and ranking_method in ["Mutual information", "MI"]:
        n_features = np.count_nonzero(ranks >= threshold)
        X_new = X[:, :n_features]
        features_new = features[:n_features]
    elif filter_method == 'Population' and 0 < threshold < 1:
        X_new = X[:, :int(threshold * X.shape[1])]
        features_new = features[:int(threshold * X.shape[1])]
    elif filter_method == 'Population' and type(threshold) == int:
        X_new = X[:, :threshold]
        features_new = features[:threshold]
    else:
        raise ValueError("No known filter method combination with threshold")

    # Return reduced matrix and feature array
    return X_new, features_new


def order_ranks(X, y, features, ranking_method="T-test", ranking_settings='Default'):
    """ Orders the matrices by rank

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The output of every sample
    :param ranking_method: The method the features will be ranking with: "T-test" (default), "Mutual information"
    :param ranking_settings: The extra settings for the ranking method: "Default" (default), test_type in t-test,
                                MI_type in mutual information
    :return: Ordered ranks
    """

    # Give all methods their ranking
    if ranking_method == "T-test":
        if ranking_settings != 'Default':
            ranks = compute_rank_ttest(X, y, ranking_settings)
        if np.unique(y).shape[0] == 2:
            ranks = compute_rank_ttest(X, y, "Unequal")
        else:
            ranks = compute_rank_ttest(X, y, "ANOVA")

        # Compute the order of the ranks
        order = np.argsort(ranks)
    elif ranking_method in ["Mutual information", "MI"]:
        ranks = compute_rank_mutual_information(X, y)
        # Compute the order of the rank, flipped, because the higher the value the better.
        order = np.flip(np.argsort(ranks), axis=0)
    else:
        raise ValueError("Unknown ranking method")

    # Order them by rank
    ordered_X = np.squeeze(X[:, order])
    ordered_features = np.squeeze(features[order])

    return ordered_X, ordered_features, ranks


def compute_rank_ttest(X, y, test_type="Unequal"):
    """ Returns the rank when using a t-test. Can be done for two groups, but also for more with the ANOVA and k
        ruskall wallis test

    :param X: X: A numpy matrix with the values for the features for every sample
    :param y: The classes of every sample
    :param test_type: The type of t-test that must be performed: "Unequal" (default), "Equal", "Paired", "ANOVA", "Kruskal"
    :return: The ranking of the t-test
    """

    # Making the arrays robust
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have a different number of samples")

    # Making the method values robust
    if test_type not in ["Unequal", "Equal", "Paired", "ANOVA", "Kruskal"]:
        raise ValueError("Unknown t-test method")
    elif test_type in ["Unequal", "Equal", "Paired"] and np.unique(y).shape[0] > 2:
        raise ValueError("More than two classes for t-test")

    # Group the values into classes
    groups = group_classes(X, y)

    # Tests
    if test_type == "Paired":
        t_value, p_value = sp.ttest_rel(groups[0], groups[1], axis=0, nan_policy='raise')
    elif test_type == "Equal":
        t_value, p_value = sp.ttest_ind(groups[0], groups[1], axis=0, nan_policy='raise')
    elif test_type == "Unequal":
        t_value, p_value = sp.ttest_ind(groups[0], groups[1], axis=0, equal_var=False, nan_policy='raise')
    elif test_type == "ANOVA":
        statistics, p_value = sp.f_oneway(*groups)
    else:
        p_value = []
        for i in range(groups[0].shape[1]):
            feature_group = []
            for group in groups:
                feature_group.append(group[:, i])
            statistics, new_p_value = sp.kruskal(*groups, axis=1, nan_policy='raise')
            p_value.append(new_p_value)
        p_value = np.asarray(p_value)

    return p_value


def compute_rank_mutual_information(X, y, test_type="Classification"):
    """ Returns the values for the features when computing mutual information with the outcome

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The output of every sample
    :param test_type: The distribution of y: "Classification" (default), "Regression"
    :return: The ranking of the mutual information computation
    """

    # Making the arrays robust
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have a different number of samples")

    if test_type == "Classification":
        MI_ranks = FS.mutual_info_classif(X, y)
    elif test_type == "Regression":
        MI_ranks = FS.mutual_info_regression(X, y)
    else:
        raise ValueError("Not the right type of Mutual Information tester")

    return MI_ranks


def group_classes(X, y):
    """ Groups the matrix of X in the classes of y

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The classes of every sample
    :return: The value matrices for every sample in a list
    """

    # Find all unique classes
    classes = np.unique(y)
    groups = []

    # Add classes as groups to a list
    for i in range(classes.shape[0]):
        groups.append(np.squeeze(X[np.argwhere(y == classes[i]), :]))

    # Return the groups
    return groups