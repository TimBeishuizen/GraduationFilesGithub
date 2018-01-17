from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from DataExtraction import DataExtraction as DE
import os
import numpy as np

""" Possible data sets:

Baseline: Normal patients (0), Non-Lesional patients (1), Lesional (2)

Psoriasis              GSE13355          180         RandomForest       NN = Normal, PN = Non-Lesional, PP = Lesional
                       GSE30999          170         KNearestNeighbour  No normal patients
                       GSE34248          28          RandomForest       No normal patients
                       GSE41662          48          LinearSVC          No normal patients
                       GSE78097          33          RandomForest       Different: Normal (0), Mild (1), Severe Psoriasis (2)
                       GSE14905          82          LinearSVC        
Atopic  dermatitis     GSE32924          33          GradientBoosting       
                       GSE27887          35          DecisionTree       Different: Pre NL (0), Post NL (1), Pre L (2), Post L (3)
                       GSE36842          39          Unknown            Also tested difference between Acute (2) and Chronic (3) Dermatitis

"""

# Extract data
print('Extracting data...')

data_names = ['GSE13355', 'GSE30999', 'GSE34248', 'GSE41662', 'GSE14905']

X = np.zeros((0,54676))
y = []
sample_ids = []

for i in range(len(data_names)):
    X_temp, y_temp, gene_ids, sample_ids_temp = DE.extract_data(data_names[i])
    X = np.append(X, X_temp, axis=0)
    y.extend(y_temp)
    sample_ids.append(sample_ids_temp)

print(X.shape)
print(len(y))

train_size = 0.9
max_opt_time = 480
pop_size = 40
n_gen = 100

# Splitting into test and training
print('Splitting into test and training...')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1 - train_size)

# Use tpot to find the best pipeline
print('Starting PipelineFinder optimization...')
tpot = TPOTClassifier(verbosity=2, max_time_mins=max_opt_time, population_size=pop_size, generations=n_gen)
tpot.fit(X_train, y_train)

# Calculate accuracy
print('The accuracy of the best pipeline is: %f' % (tpot.score(X_test, y_test)))

# Export pipeline
print('Exporting as TPOT_' + 'psoriasis' + '_pipeline.py')
cwd = os.getcwd()
os.chdir('../Pipelines')
tpot.export('TPOT_' + 'psoriasis' + '_pipeline.py')
os.chdir(cwd)