from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from DataExtraction import DataExtraction as DE
import os


def select_pipeline_tpot(data_name, train_size, max_opt_time, n_gen, pop_size):
    """ Selects the best pipeline with tpot and exports its file

    :param data_name: Name of the data
    :param train_size: The sizes of the training and test set, in a fraction of the complete set
    :param max_opt_time: The maximal optimization time for the tpot classifier
    :param n_gen: The number of generations used in the tpot classifier
    :param pop_size: The population size used in the tpot classifier
    :return: an exported python file containing the best pipeline
    """

    # Extract data
    print('Extracting data...')
    X, y, gene_ids, sample_ids = DE.extract_data(data_name)

    # Splitting into test and training
    print('Splitting into test and training...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size)

    # Use tpot to find the best pipeline
    print('Starting PipelineFinder optimization...')
    tpot = TPOTClassifier(verbosity=2, max_time_mins=max_opt_time, population_size=pop_size, generations=n_gen)
    tpot.fit(X_train, y_train)

    # Calculate accuracy
    print('The accuracy of the best pipeline is: %f' % (tpot.score(X_test, y_test)))

    # Export pipeline
    print('Exporting as TPOT_' + data_name + '_pipeline.py')
    cwd = os.getcwd()
    os.chdir('../Pipelines')
    tpot.export('TPOT_' + data_name + '_pipeline.py')
    os.chdir(cwd)