from sklearn.model_selection import train_test_split
from DataExtraction import DataExtraction as DE
import os
from sklearn import tree as T
import numpy as np
import csv
from PipelineEvaluation import PipelineFileGetter as PFG


def evaluate_pipeline(data_name, train_size):
    """ Evaluate the pipeline to find out how good it works

    :param data_name: The name of the data set
    :param train_size: The size of the train and test part of the data set
    :return: Evaluation of the pipelines
    """

    # Extract data
    print('Extracting data...')
    X, y, gene_ids, sample_ids = DE.extract_data(data_name)

    # Splitting into test and training
    print('Splitting into test and training...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size)

    # Use tpot to find the best pipeline
    print('Fit data on best pipeline...')
    pipeline = PFG.find_correct_pipeline(data_name)
    pipeline.fit(X_train, y_train)

    # Calculate accuracy
    print('The accuracy of the best pipeline is: %f' % (pipeline.score(X_test, y_test)))

    # Create tree
    cwd = os.getcwd()
    os.chdir('../Pictures')

    if 'randomforestclassifier' in pipeline.named_steps:
        # Show one tree of the forest
        tree = pipeline.named_steps['randomforestclassifier'].estimators_[0]
        T.export_graphviz(tree, out_file='tree_' + data_name + '.dot', feature_names=gene_ids, filled=True, rounded=True)
    elif 'linearsvc' in pipeline.named_steps:
        # show the best 25 coefficients
        coeff = pipeline.named_steps['linearsvc'].coef_

        max_coeff = np.max(np.absolute(coeff[:]), 0)
        sort_coeff = np.argsort(max_coeff)

        cw = csv.writer(open('linearsvcCoeff' + data_name + '.csv', 'w'))
        cw.writerow(np.array(gene_ids)[sort_coeff[-26:-1]])
        for i in range(coeff.shape[0]):
            cw.writerow(coeff[i, sort_coeff[-26:-1]])
    elif 'decisiontreeclassifier' in pipeline.named_steps:
        # Show the tree of the forest
        tree = pipeline.named_steps['decisiontreeclassifier']
        T.export_graphviz(tree, out_file='tree_' + data_name + '.dot', feature_names=gene_ids, filled=True,
                          rounded=True)
    elif 'gradientboostingclassifier' in pipeline.named_steps:
        # Show one tree of the forest
        tree = pipeline.named_steps['gradientboostingclassifier'].estimators_[1, 0]
        T.export_graphviz(tree, out_file='tree_' + data_name + '.dot', feature_names=gene_ids, filled=True,
                          rounded=True)

    else:
        print(pipeline.named_steps)

    os.chdir(cwd)