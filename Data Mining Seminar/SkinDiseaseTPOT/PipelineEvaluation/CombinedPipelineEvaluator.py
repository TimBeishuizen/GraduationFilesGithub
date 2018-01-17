from sklearn import ensemble as E
from sklearn import linear_model as LM
from sklearn import pipeline as PL
import numpy as np
from sklearn.model_selection import train_test_split
from DataExtraction import DataExtraction as DE
from PipelineEvaluation import PipelineFileGetter as PFG

# Extract data
print('Extracting data...')

psoriasis_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905']
data_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE78097','GSE14905','GSE32924','GSE27887']

X = np.zeros((0,54676))
y = []
sample_ids = []

for i in range(len(psoriasis_names)):
    X_temp, y_temp, gene_ids, sample_ids_temp = DE.extract_data(psoriasis_names[i])
    X = np.append(X, X_temp, axis=0)
    y.extend(y_temp)
    sample_ids.append(sample_ids_temp)

train_size = 0.9


def find_combined_pipeline_score(X, y, train_size):
    """ Find the socre of the data set for the combined pipeline

    :param X: the data matrix
    :param y: the output labels
    :param train_size: size of train and test data set
    :return: score of 10 different testing times
    """
    score = []

    # Pipeline combined data set
    for i in range(10):
        print('Currently at pipeline %i' % i)
        pipeline = PL.make_pipeline(LM.LogisticRegression(C=0.0001, dual=True, penalty='l2'),
                                    E.GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=0.05,
                                                                 min_samples_leaf=12,
                                                                 min_samples_split=19, n_estimators=100, subsample=0.5))

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1 - train_size)
        pipeline.fit(X_train, y_train)
        score.append(pipeline.score(X_test, y_test))

    return score


def evaluate_combined_pipeline_score(X, y, data_name, train_size):
    """ Evaluate the scores for the combined data set for different pipelines

    :param X: the data matrix
    :param y: the output labels
    :param data_name: name of the data set the pipeline was generated from
    :param train_size: the size of the train and test data set
    :return: score of 10 different testing times
    """

    score = []

    # Pipeline combined data set
    for i in range(10):
        print('Currently at pipeline %i' % i)
        pipeline = PFG.find_correct_pipeline(data_name)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1 - train_size)
        pipeline.fit(X_train, y_train)
        score.append(pipeline.score(X_test, y_test))

    return score


# c_score = find_combined_pipeline_score(X, y, train_size)
# print("The average score for the combined pipeline is %f" % (sum(c_score)/len(c_score))


final_score = []

for data_name in data_names:
    print('Currently evaluating pipeline %s' % data_name)
    p_score = evaluate_combined_pipeline_score(X, y, data_name, train_size)
    final_score.append(sum(p_score)/len(p_score))
    print(final_score[-1])

print('Final scores:')
print(data_names)
print(final_score)

