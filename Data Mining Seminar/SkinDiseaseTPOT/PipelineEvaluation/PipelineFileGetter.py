from sklearn import ensemble as E
from sklearn import preprocessing as PP
from sklearn import pipeline as PL
from sklearn import svm
from sklearn import tree as T
from sklearn import neighbors as N

def find_correct_pipeline(data_name):
    """ Returns the best pipeline for the data set

    :param data_name: The name for the data set
    :return: The used pipeline
    """

    if data_name == 'GSE13355':
        pipeline = PL.make_pipeline(
            PP.Binarizer(threshold=0.7000000000000001),
            E.RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.35000000000000003,
                                     min_samples_leaf=3, min_samples_split=18, n_estimators=100)
        )
    elif data_name == 'GSE14905':
        pipeline = PL.make_pipeline(svm.LinearSVC(C=5.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.001))
    elif data_name == 'GSE27887':
        pipeline = PL.make_pipeline(T.DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=1, min_samples_split=10))
    elif data_name == 'GSE30999':
        pipeline = PL.make_pipeline(N.KNeighborsClassifier(n_neighbors=9, p=2, weights="distance"))
    elif data_name == 'GSE32924':
        pipeline = PL.make_pipeline(E.GradientBoostingClassifier(learning_rate=1.0, max_depth=8, max_features=0.7500000000000001, min_samples_leaf=4,
                                   min_samples_split=3, n_estimators=100, subsample=0.7000000000000001))

    elif data_name == 'GSE34248':
        pipeline = PL.make_pipeline(E.RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.9000000000000001, min_samples_leaf=3,
                               min_samples_split=6, n_estimators=100))
    elif data_name == 'GSE41662':
        pipeline = PL.make_pipeline(svm.LinearSVC(C=0.001, dual=True, loss="hinge", penalty="l2", tol=0.01))
    elif data_name == 'GSE78097':
        pipeline = PL.make_pipeline(
            E.RandomForestClassifier(bootstrap=False, criterion="gini", max_features=1.0, min_samples_leaf=4,
                                     min_samples_split=10, n_estimators=100))
    elif data_name == 'GSE36842':
        raise NotImplementedError()
    else:
        raise NotImplementedError('No pipeline is created for this data set')

    return pipeline
