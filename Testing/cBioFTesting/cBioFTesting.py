import sys
sys.path.append(r"C:\Users\s119104\Documents\GitHub\cBioF")

from DataExtraction import DataSetExtraction
import numpy as np

X, y, features = DataSetExtraction.import_example_data('Psoriasis')

print(np.count_nonzero(y == 0))
print(np.count_nonzero(y == 1))
print(np.count_nonzero(y == 2))

print(y.shape)