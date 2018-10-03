from metalearn import Metafeatures
from DataExtraction import DataSetExtraction as DSE
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

X, y, features = DSE.import_example_data('MicroOrganisms')

missing_values = ''

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)

print(data.shape)
print(X[:, 1].shape)

plt.boxplot(X[:, 10].astype(float))
plt.show()

