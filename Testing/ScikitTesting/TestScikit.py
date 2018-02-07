from FileExtraction import PreprocessData as PD
from ScikitTesting import HotEncodeData as HED
from sklearn import svm
import numpy as np

# Obtain the data
train_strands, train_names, train_outcome = PD.create_combination_gene_file('train', 'acceptor')
test_strands, test_names, test_outcome = PD.create_combination_gene_file('test', 'acceptor')

# Hot encode the data
train_X = HED.hot_encode_data(train_strands)
test_X = HED.hot_encode_data(test_strands)

clf = svm.SVC()
clf.fit(train_X, train_outcome)
test_Y = clf.predict(test_X)
false_XOR = np.logical_xor(test_outcome, test_Y)
print((np.size(false_XOR) - np.count_nonzero(false_XOR))/np.size(false_XOR))