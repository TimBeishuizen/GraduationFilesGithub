from DataExtraction import DataSetExtraction as DSE
from DataExplorationMethods.FindDatasetIssuesMethods import find_dataset_issues as fdi

X, y, features = DSE.import_example_data('Hepatitis')

missing_values = ''

X, y, features, exploration_results = fdi(X, y, features, missing_values=missing_values, focus=True, preprocessing=True)#, plots=True)

print(features)