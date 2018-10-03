import numpy as np

from MissingValueHandlingMethods import SingleImputationMethods as SIM


def MICE(X, missing_values=None, s=1, m=1):
    """

    :param X:
    :param y:
    :param missing_values:
    :param imputation_value:
    :return:
    """

    X = np.copy(np.asarray(X))

    new_Xs = []

    # Create multiple datasets
    for i in range(m):
        print("Currently at imputation dataset %i" % i)
        # Create a dataset with chained equations
        new_X = chained_equations(X, missing_values=missing_values, s=s)
        new_Xs.append(np.asarray(new_X))

    return new_Xs


def chained_equations(X, missing_values=None, s=1):
    """

    :param X:
    :param y:
    :param missing_values:
    :param s:
    :return:
    """

    X = np.array(X)

    # Find locations of the missing values
    missing_loc = np.argwhere(X == missing_values)
    missing_feat = np.unique(missing_loc[:, 1])

    # Make a list for every feature for their missing values
    missing_feat_loc = []
    for i in range(X.shape[1]):
        missing_feat_loc.append([])

    for loc in missing_loc:
        missing_feat_loc[loc[1]].append(loc)

    new_X = SIM.mean_imputation(X, missing_values=missing_values)

    for i in range(s):
        print("\t at cycle %i" % i)
        # Create new order to replace features
        np.random.shuffle(missing_feat)

        # Replace missing values per feature
        for feat in missing_feat:
            #print("\t\t at feature %i" % feat)

            # Remove imputed values for the feature
            for loc in missing_feat_loc[feat]:
                new_X[loc[0], loc[1]] = ''

            # Impute them with regression again
            new_X = SIM.regression_imputation(new_X, missing_values=missing_values)

    return new_X
