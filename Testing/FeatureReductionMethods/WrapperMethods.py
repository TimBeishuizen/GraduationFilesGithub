import numpy as np
import random

from FeatureReductionMethods import FilterMethods as FM

from sklearn import svm as SVM
from sklearn import model_selection as MS
from sklearn import tree as T
from sklearn import naive_bayes as NB

def wrapper_methods_classifier():
    return None


def sequential_search(X, y, features, sel_seq='F', n_sel_seq=10, scoring_method='svm', scoring_cv=10,
                      improvement_threshold=0.01, ranking_method=None, max_iter=100, continued_search=False):
    """

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The classes of every sample
    :param features: The names of the features
    :param sel_seq: The selection of sequential search. Can be string or list of strings:
                        'F' (forward selection, default)
                        'B' (backward selection)
                        'FB' (forward then backward selection)
                        'BF' (backward then forward selection)
    :param n_sel_seq: The number of features that should be selected by the sel_seq algorithm
    :param scoring_method: The method to be used to select features for adding to the subset
    :param scoring_cv: The number of cross validations done in the scoring method
    :param improvement_threshold: The threshold to be used to select features for adding to the subset
    :param ranking_method: The method of ordering the features if needed
    :param continued_search: Gives the possibility to continue search within parameters instead of starting anew
    :return: A subset of X and features
    """

    # Check whether matrix parameters are correctly sized
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y do not match")
    elif X.shape[1] != features.shape[0]:
        raise ValueError("The number of features in X and the array of features do not match")

    # Check other parameters
    if not (0 < improvement_threshold < 1):
        raise ValueError("Improvement threshold not in predefined range <0, 1>")
    elif not type(continued_search) == bool:
        raise ValueError("Continued search is not a boolean (True or False)")

    # Split selection sequences
    if type(sel_seq) == str:
        sel_seq = list(sel_seq)
    if type(n_sel_seq) == int:
        n_sel_seq = [n_sel_seq]

    # Check length selection algorithms
    if len(sel_seq) != len(n_sel_seq):
        raise ValueError("The number of sequence selection algorithms and the ")

    # Check names in selection algorithms
    for i in range(len(sel_seq)):
        if sel_seq[i] not in ['F', 'B']:
            raise ValueError('A sequence selection algorithm not in known algorithms')
        if not (X.shape[1] > n_sel_seq[i] > 0) or type(n_sel_seq[i]) != int:
            raise ValueError('Wrong number threshold selectors for the sequence selection algorithm')

    print("Ordering features...")
    ordered_X, ordered_features = order_features(X, y, features, ranking_method=ranking_method)

    # Initialize for first search
    if sel_seq[0] == 'F':
        candidate_features = np.array([])
        candidate_X = np.zeros((ordered_X.shape[0], 0))
    elif sel_seq[0] == 'B':
        candidate_features = ordered_features
        candidate_X = ordered_X
    else:
        raise ValueError("No possible search algorithm")

    # Create dummy new_features, so it will always iterate + start iteration threshold
    new_features = np.array(['Dummy'])
    old_features = np.array(['Dummy2'])
    iteration = 0
    new_search = 0

    # As long as the candidate features do not change
    while set(new_features) != set(candidate_features) != set(old_features) and iteration < max_iter and \
                    new_search < features.shape[0]:
        iteration += 1
        print("Currently at iteration %i" % iteration)

        # Keep track of old feature sets
        old_features = new_features
        new_features = candidate_features

        # Do all Forward and backward steps
        for i in range(len(sel_seq)):
            if sel_seq[i] == 'F':
                candidate_X, candidate_features, new_search = forward_selection(ordered_X, y, ordered_features,
                                                                    scoring_method=scoring_method,
                                                                    improvement_threshold=improvement_threshold,
                                                                    number_threshold=n_sel_seq[i], base_X=candidate_X,
                                                                    base_features=candidate_features, scoring_cv=scoring_cv,
                                                                                continued_search=new_search)
            elif sel_seq[i] == 'B':
                candidate_X, candidate_features = backward_selection(candidate_X, y, candidate_features,
                                                                     scoring_method=scoring_method,
                                                                     improvement_threshold=improvement_threshold,
                                                                     number_threshold=n_sel_seq[i], scoring_cv=scoring_cv)
            else:
                raise ValueError("No possible search algorithm")

            if continued_search is False:
                new_search = 0


    return candidate_X, candidate_features


def forward_selection(X, y, features, scoring_method="svm", improvement_threshold=0.01,
                      number_threshold=None, base_X=None, base_features=None, ranking_method=None, scoring_cv=10,
                      continued_search=0):
    """

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The classes of every sample
    :param features: The names of the features
    :param scoring_method: The method to be used to select features for adding to the subset
    :param scoring_cv: The number of cross validations to be done for threshold
    :param improvement_threshold: The threshold to be used to select features for adding to the subset
    :param number_threshold: The maximum number of features that can be added
    :param base_X: The initial matrix with the values of the features already in the set
    :param base_features: The names of the features already in the set
    :param ranking_method: The method of ordering the features if needed
    :param continued_search: A possible continuation of the search location when doing multiple forward selection:
                            default 0 (begin at start)
    :return: A subset of X and features
    """

    # Check whether matrix parameters are correctly sized
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y do not match")
    elif X.shape[1] != features.shape[0]:
        raise ValueError("The number of features in X and the array of features do not match")
    elif (base_X is not None) and not (base_features is not None):
        raise ValueError("Both the base of the matrix as the base of the features should be present, not either")
    elif base_X is not None and base_features is not None and base_X.shape[1] != base_features.shape[0]:
        raise ValueError("The number of features in the base of X and the base array of features do not match")

    # Define number threshold if not previously defined
    if number_threshold is None:
        number_threshold = features.shape[0] - 1

    # Check other parameters
    if not (0 < improvement_threshold < 1):
        raise ValueError("Improvement threshold not in predefined range <0, 1>")
    elif not (0 < number_threshold < X.shape[1]):
        raise ValueError("Number threshold is not in range [0, %i]" % X.shape[1])
    elif type(number_threshold) is not int:
        raise ValueError("Number threshold is not an integer value")
    elif (not 0 <= continued_search < features.shape[0]) or not type(continued_search) == int:
        raise ValueError("The continuation of the search was from a point not between the number of features")

    # Order X if needed
    X, features = order_features(X, y, features, ranking_method=ranking_method)

    # Initial start of the selected features
    if base_X is None:
        selected_X = np.zeros((X.shape[0], 0))
        selected_features = np.array([])
        score = 0
    else:
        selected_X = base_X
        selected_features = base_features
        if selected_X.shape[1] != 0:
            score = compute_score(base_X, y, scoring_method, cv=scoring_cv)
        else:
            score = 0

    # Initialize running criteria
    max_features = selected_features.shape[0] + number_threshold

    print("Starting forward selection with scoring %f, starting from feature %i..." % (score, continued_search))
    # Use forward selection
    for i in range(continued_search, features.shape[0]):

        continued_search = i + 1

        if features[i] in selected_features.tolist():
            continue

        # Create a candidate feature set by adding new feature
        candidate_X = np.concatenate((selected_X, X[:, i:i+1]), axis=1)
        new_score = compute_score(candidate_X, y, scoring_method=scoring_method, cv=scoring_cv)

        # Add new feature if new score is better
        if new_score - score > improvement_threshold:
            # print('Selected feature %s, with improved score %f' % (features[i], float(new_score)))
            selected_X = candidate_X
            selected_features = np.append(selected_features, features[i])
            score = new_score

        # Stop if limit has been reached
        if selected_features.shape[0] >= np.min([max_features, features.shape[0]]):
            continued_search = i
            break

        # if i%int(features.shape[0]/10) == 0:
            # print("Currently done %i of the %i features" % (i, features.shape[0]))

    return selected_X, selected_features, continued_search


def backward_selection(X, y, features, scoring_method="svm", improvement_threshold=0.001,
                      number_threshold=None, ranking_method=None, scoring_cv=10):
    """ Performs feature selection by selection by removing features from the original dataset, using the scoring method
        and the improvement method to selected the features for removal. If needed the features are ordered by
        ranking method

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The classes of every sample
    :param features: The names of the features
    :param scoring_method: The method to be used to select features for removal
    :param scoring_cv: The number of cross validations to be done: default = 10
    :param improvement_threshold: The threshold to be used to select features for removal
    :param number_threshold: The maximum number of features that can be removed
    :param ranking_method: The method of ordering the features if needed
    :return: A subset of X and features
    """

    # Check whether matrix parameters are correctly sized
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y do not match")
    elif X.shape[1] != features.shape[0]:
        raise ValueError("The number of features in X and the array of features do not match")

    # Define number threshold if not previously defined
    if number_threshold is None:
        number_threshold = 1

    # Check other parameters
    if not (0 < improvement_threshold < 1):
        raise ValueError("Improvement threshold not in predefined range <0, 1>")
    elif not (0 < number_threshold < X.shape[1]):
        print("Number threshold is not in range [0, %i]" % X.shape[1])
    elif type(number_threshold) is not int:
        raise ValueError("Number threshold is not an integer value")

    # Order X if needed
    X, features = order_features(X, y, features, ranking_method=ranking_method)

    # Initialize running criteria
    min_features = features.shape[0] - number_threshold
    score = compute_score(X, y, scoring_method, cv=scoring_cv)
    selected_X = X
    selected_features = features

    print("Starting backward selection with score %f..." % score)
    # Use backward selection - starting in the back
    for i in list(reversed(range(features.shape[0]))):
        # Create a candidate feature set by removing feature
        candidate_X = np.delete(selected_X, i, axis=1)
        new_score = compute_score(candidate_X, y, scoring_method=scoring_method, cv=scoring_cv)

        # Remove feature if new score is not much worse
        if score - new_score < improvement_threshold:
            # print('Removed feature %s with new score %f' % (selected_features[i], float(new_score)))
            selected_X = candidate_X
            selected_features = np.delete(selected_features, i, axis=0)
            score = new_score

        # Stop if limit has been reached
        if selected_features.shape[0] <= np.min([0, min_features]):
            break

    return selected_X, selected_features


def order_features(X, y, features, ranking_method=None):
    """ Order the features by means of ranking method

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The classes of the samples
    :param features: The name of the features for every sample
    :param ranking_method: The way of ranking the features: None (no ranking, default), 'Random' (randomly), 'T-test',
                            'Mutual information'
    :return: Ordered X and features
    """

    # The possible rankings used for ordering
    if ranking_method == 'Random':
        order = random.sample(range(features.shape[0]), features.shape[0])
        ordered_X = X[:, order]
        ordered_features = features[order]
    elif ranking_method in ["T-test", "Mutual information", "MI"]:
        ordered_X, ordered_features, _ = FM.order_ranks(X, y, features, ranking_method=ranking_method)
    elif ranking_method is None:
        ordered_X = X
        ordered_features = features
    else:
        raise ValueError("Unknown ranking method")

    return ordered_X, ordered_features


def compute_score(X, y, scoring_method="svm", cv=10):
    """ Computes the score of X predicting y

    :param X: A numpy matrix with the values for the features for every sample
    :param y: The classes of every sample
    :param scoring_method: The way of scoring the quality of the new set
    :param cv: The number of cross validation rounds to be used: 100 'default
    :return: the score of X's ability to predict y
    """

    if scoring_method == "svm":
        alg = SVM.LinearSVC()
    elif scoring_method == 'dt':
        alg = T.DecisionTreeClassifier()
    elif scoring_method == 'nb':
        alg = NB.GaussianNB()
    else:
        raise ValueError("No such scoring method known")

    return np.mean(MS.cross_val_score(alg, X, y, cv=cv))

