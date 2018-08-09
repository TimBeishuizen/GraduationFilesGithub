import numpy as np
import sklearn.linear_model as SLM
import sklearn.model_selection as SMS
import sklearn.neighbors as SN
import sklearn.svm as SVM
import sklearn.tree as ST
import sklearn.naive_bayes as NB
from sklearn import metrics as SME

from FeatureReductionMethods import FilterMethods as FM


def test_filter_methods_classifier(X, y, features, filter_values, ranking_method = 'T-test', filter_method = 'rank',
                                   ML_algorithm = 'svm'):
    """ Tests the filter methods classifier for its ability to correctly reduce features

    :param X: The matrix with the values for the classification problem (sample x features)
    :param y: The classes for every sample
    :param features: The tested features
    :param filter_values: The values for the filtering that were used
    :param rank_type: The type of ranking to be used used
    :param filter_type: The type of filtering that is used
    :param ML_algorithm: The machine learning algorithm to be used
    :return: The number of features left, joined with the validation and test score.
    """

    # Testing regular score without classifiers
    # print("Testing without feature selection...")
    # val_reg, test_reg, prec, rec, Fbeta = test_method(X, y, ML_algorithm=ML_algorithm)
    # print("Validation score: %f, Test score: %f" % (float(val_reg), test_reg))

    # Initializing values for number of features and the validation and test score
    # val_score = [val_reg]
    # test_score = [test_reg]
    # prec_score = [prec]
    # rec_score = [rec]
    # Fbeta_score = [Fbeta]
    # feat = [X.shape[1]]

    val_score = []
    test_score = []
    prec_score = []
    rec_score = []
    Fbeta_score = []
    feat = []


    # Testing after feature selection
    for value in filter_values:
        print("Filtering with ranking method %s and filter method %s on value %f" % (ranking_method, filter_method, value))

        # Feature selection
        X_new, features_new, _ = FM.filter_methods_classifier(X, y, features, ranking_method=ranking_method,
                                                           filter_method=filter_method,
                                                           threshold=value)

        # Testing after feature selection
        new_val, new_test, prec, rec, Fbeta = test_method(X_new, y, ML_algorithm=ML_algorithm)
        val_score.append(new_val)
        test_score.append(new_test)
        prec_score.append(prec)
        rec_score.append(rec)
        Fbeta_score.append(Fbeta)
        feat.append(X_new.shape[1])
        print("Features left: %i, Validation score: %f, Test score: %f" % (X_new.shape[1], float(new_val), new_test))

    # Return the number of features and the validation and test score
    return val_score, test_score, feat, prec_score, rec_score, Fbeta_score



def test_method(X, y, ML_algorithm = 'svm'):
    """ Tests a method by using Leave One Out validation of a machine learning classifier.

    :param X: The values for every sample and feature
    :param y: The output values for every sample
    :param ML_algorithm: The machine learning algorithm used for testing: "svm" (support vector machines, default),
                        "dt" (decision tree), "nn" (Nearest Neighbour), "lg" (Logistic Regression)
    :return: The validation and the test score
    """

    # Split te data in a train and a test set
    X_train, X_test, y_train, y_test = SMS.train_test_split(X, y, train_size=0.8)

    # Initalize a leave one out and a validation score
    loo = SMS.LeaveOneOut()
    val_score = []

    # Small method for the machine learning algorithm
    def choose_ml(ML_algorithm):
        if ML_algorithm == 'svm':
            return SVM.LinearSVC()
        elif ML_algorithm == 'dt':
            return ST.DecisionTreeClassifier()
        elif ML_algorithm == 'nn':
            return SN.KNeighborsClassifier()
        elif ML_algorithm == 'lr':
            return SLM.LogisticRegression()
        elif ML_algorithm == 'nb':
            return NB.GaussianNB()

    # Compute validation score
    for train_index, test_index in loo.split((X_train)):
        ml = choose_ml(ML_algorithm)
        ml.fit(X_train[train_index], y_train[train_index])

        val_score.append(ml.score(X_train[test_index], y_train[test_index]))

    # Compute the mean of the validation score
    val_score = np.mean(np.asarray(val_score))

    # Compute test score
    ml = choose_ml(ML_algorithm)
    ml.fit(X_train, y_train)
    test_score = ml.score(X_test, y_test)

    y_pred = ml.predict(X_test)
    prec, rec, Fbeta, _ = SME.precision_recall_fscore_support(y_test, y_pred, average='weighted')

    return val_score, test_score, prec, rec, Fbeta
