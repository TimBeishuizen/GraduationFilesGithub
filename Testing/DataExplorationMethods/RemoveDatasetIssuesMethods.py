import numpy as np
from sklearn.feature_selection import SelectFwe as SF, f_classif
from sklearn.preprocessing import StandardScaler as SS, MinMaxScaler as MMS, MultiLabelBinarizer as MLB

from DataExplorationMethods import wrapper_methods as WM, order_methods as OM
from MissingValueHandlingMethods import ListDeletionMethods as LDM, SingleImputationMethods as SIM


def preprocess_dataset(dfX, dfy, exploration_results):
    """

    :param dfX:
    :param dfy:
    :param exploration_results:
    :return:
    """

    dfX = dfX.replace(np.NaN, '')

    X = dfX.values
    y = dfy.cat.codes.values
    features = list(dfX)

    # First change data for missing values
    if exploration_results['mv']:
        old_features = np.copy(features)
        if exploration_results['cca']:
            X, y = LDM.cca(X, y, missing_values='')
        elif exploration_results['aca']:
            X, features = LDM.aca(X, features, missing_values='')
        else:
            X, features = LDM.aca(X, features, missing_values='', removal_fraction=0.15)
            X = SIM.mean_imputation(X, missing_values='')

        removed_features = return_removed_features(features, old_features)

        print("These features are removed due to having too many missing values: %s" % removed_features)


    if exploration_results['irrelevance'] > 0:
        # Remove irrelevant
        irr_feat_loc = exploration_results['irrelevant_features']
        X = np.delete(X, irr_feat_loc, axis=1)
        old_features = np.copy(features)
        features = np.delete(features, irr_feat_loc)
        removed_features = return_removed_features(features, old_features)

        print("These features are removed due to having no information: %s" % removed_features)

        return_removed_features(features, old_features)

    if exploration_results['norm_means'] or exploration_results['norm_stdev']:
        # Normalise or standardise values
        normalise_numeric_features(X, exploration_results['stand'],
                                   exploration_results['norm_means'], exploration_results['norm_stdev'])

    # Than change categorical to numeric values
    if exploration_results['cat']:
        X, features = hot_encode_categorical_features(X, features)

    if exploration_results['fs']:
        # Feature selection if multicollinearity
        if exploration_results['mc'] and False:
            # Remove multicollinearity
            feature_selector = WM.ForwardSelector(threshold=0.001)

            # Order to have more relevant features first
            feature_orderer = OM.FeatureOrderer(f_classif)
            X = feature_orderer.fit_transform(X, y)
            features = features[np.argsort(-feature_orderer.scores_)]
        else:
            feature_selector = SF(f_classif, alpha=0.05)

        X = feature_selector.fit_transform(X, y)
        old_features = np.copy(features)
        features = features[feature_selector.get_support()]
        removed_features = return_removed_features(features, old_features)

        print("These features are removed due to feature selection: %s" % removed_features)

    if exploration_results['imbalance']:
        print("Try to use F1-score over Accuracy in quality measurements")
        # F1 score

    return X, y, features


def column_types_dataset(X, categorical=True):
    """

    :param dfX:
    :return:
    """

    column_types = []

    def dtype_is_categorical(dtype, categorical=True):
        if categorical:
            return not("int" in str(dtype) or "float" in str(dtype))
        else:
            return ("int" in str(dtype) or "float" in str(dtype))

    for i in range(X.shape[1]):
        column_types.append(dtype_is_categorical(type(X[0, i]), categorical))

    return column_types


def hot_encode_categorical_features(X, features):
    """

    :param X:
    :param features:
    :return:
    """

    column_types = column_types_dataset(X)

    X_new = np.zeros((X.shape[0], 0))
    f_new = []

    for i in range(len(column_types)):
        if column_types[i]:
            # Hot encode categories
            hot_encoder = MLB()
            new_col = hot_encoder.fit_transform(X[:, i:i+1])

            X_new = np.append(X_new, new_col, axis=1)
            for label in hot_encoder.classes_:
                f_new.append(features[i] + '_' + label)
        else:
            X_new = np.append(X_new, X[:, i:i+1], axis=1)
            f_new.append(features[i])

    return X_new, np.asarray(f_new)


def normalise_numeric_features(X, standardisation=False, means=True, stdev=True):
    """

    :param X:
    :param standardisation:
    :param means:
    :param stdev:
    :return:
    """

    column_types = column_types_dataset(X, categorical=False)

    for i in range(len(column_types)):
        if column_types[i]:

            if standardisation:
                # Standardisation
                scaler = MMS([0, 1])
                X[:, i:i+1] = scaler.fit_transform(X[:, i:i+1])
            else:
                # Normalisation
                scaler = SS(means, stdev)
                X[:, i:i+1] = scaler.fit_transform(X[:, i:i+1])

    return X


def return_removed_features(new_features, old_features):
    """

    :param new_features:
    :param old_features:
    :return:
    """

    # Show removed features
    removed_features = []
    for feat in old_features:
        if feat not in new_features:
            removed_features.append(feat)

    return removed_features