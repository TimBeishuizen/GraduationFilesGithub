from sklearn import preprocessing as pp
import numpy as np


def label_encode_feature(feature_data):
    """ Label encodes a features
    
    :param feature_data: the data of the specific feature
    :return: A label encoding of the feature
    """

    # Create label encoder and encode
    le = pp.LabelEncoder()
    le.fit(['A', 'T', 'G', 'C'])
    transformed = le.transform(feature_data)

    return transformed


def hot_encode_features(features_data):
    """ Label encodes features

    :param features_data: the data of the specific features
    :return: A hot encoding of the features
    """

    # Create hot encoder and encode
    OHE = pp.OneHotEncoder()
    OHE.fit(features_data)
    encoded = OHE.transform(features_data).toarray()

    return encoded


def hot_encode_data(data):
    """ Hot encodes the data for every feature
    
    :param data: Data containing multiple features
    :return: A hot encoding for multiple features
    """

    # Create dummy data for the labeled features
    labeled_data = np.zeros(data.shape)

    # Go through ever feature to label them
    for i in range(data.shape[1]):
        feature_data = label_encode_feature(data[:, i])
        labeled_data[:, i] = feature_data

    # Encode every label
    encoded_data = hot_encode_features(labeled_data)

    return encoded_data