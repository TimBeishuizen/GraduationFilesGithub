from tpot import TPOTClassifier, HighFSTPOTClassifier
from sklearn.model_selection import train_test_split
from DataExtraction import DataSetExtraction as DSE
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
    X, y, features = DSE.import_example_data('MicroOrganisms')

    # Splitting into test and training
    print('Splitting into test and training...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size)

    # Use tpot to find the best pipeline

    tpot = HighFSTPOTClassifier(verbosity=2, max_time_mins=max_opt_time, population_size=pop_size, generations=n_gen,
                          feature_selection=True, fs_modifier=0.001)
    print('Starting PipelineFinder optimization...')
    tpot.fit(X_train, y_train)

    # Calculate accuracy
    print('The accuracy of the best pipeline is: %f' % (tpot.score(X_test, y_test)))

    # Export pipeline
    print('Exporting as TPOT_' + data_name + '_pipeline.py')
    cwd = os.getcwd()
    os.chdir('../Pipelines')
    tpot.export('TPOT_' + data_name + '_pipeline.py')
    os.chdir(cwd)


def create_experiment_tpot(data_names, selection_types, algorithm_types, train_size, max_opt_time, n_gen, pop_size, ):
    """ Starts an experiment that shows the quality of the new outcome

        :param data_names: Names of the data
        :param selection_types:
        :param algorithm_types:
        :param train_size: The sizes of the training and test set, in a fraction of the complete set
        :param max_opt_time: The maximal optimization time for the tpot classifier
        :param n_gen: The number of generations used in the tpot classifier
        :param pop_size: The population size used in the tpot classifier
        :return: an exported python file containing the best pipeline
        """

    for data_name in data_names:

        # Extract data
        print('Extracting %s data...' % data_name)
        X, y, features = DSE.import_example_data(data_name)

        for selection_type in selection_types:
            for algorithm_type in algorithm_types:
                # Start experiment
                print('Starting with %s and %s experiment...' % (selection_type, algorithm_type))

                # Splitting into test and training
                print('Splitting into test and training...')
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1 - train_size)

                try:
                    # Use tpot to find the best pipeline
                    if selection_type == 'regular_selection' and algorithm_type == 'regular_algorithms':
                        tpot = TPOTClassifier(verbosity=2, max_time_mins=max_opt_time, population_size=pop_size, generations=n_gen,
                                          feature_selection=False, fs_modifier=0.001)
                    elif selection_type == 'regular_selection' and algorithm_type == 'FS_algorithms':
                        tpot = HighFSTPOTClassifier(verbosity=2, max_time_mins=max_opt_time, population_size=pop_size,
                                                generations=n_gen,
                                                feature_selection=False, fs_modifier=0.001)
                    elif selection_type == 'FS_selection' and algorithm_type == 'regular_algorithms':
                        tpot = TPOTClassifier(verbosity=2, max_time_mins=max_opt_time, population_size=pop_size,
                                          generations=n_gen,
                                          feature_selection=True, fs_modifier=0.001)
                    elif selection_type == 'FS_selection' and algorithm_type == 'FS_algorithms':
                        tpot = HighFSTPOTClassifier(verbosity=2, max_time_mins=max_opt_time, population_size=pop_size,
                                                generations=n_gen,
                                                feature_selection=True, fs_modifier=0.001)
                    else:
                        raise ValueError('incorrect selection or algorithm types')

                except:
                    print("An error occurred. Continuing with next test.")

                print('Starting PipelineFinder optimization...')
                tpot.fit(X_train, y_train)

                # Calculate accuracy
                print('The accuracy of the best pipeline is: %f' % (tpot.score(X_test, y_test)))

                # Export pipeline
                print('Exporting as TPOT_' + data_name + '_pipeline.py')
                cwd = os.getcwd()
                os.chdir('../Pipelines')
                tpot.export('TPOT_' + data_name + '_' + selection_type + '_' + algorithm_type + '_pipeline.py')
                os.chdir(cwd)