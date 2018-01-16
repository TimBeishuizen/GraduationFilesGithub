from PipelineEvaluation import PipelineEvaluator as PE
from PipelineEvaluation import CrossPipelineEvaluator as CPE
import numpy as np
import matplotlib.pyplot as plt

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

# PE.evaluate_pipeline('GSE13355', 0.9)
# PE..evaluate_pipeline('GSE14905', 0.9)
# PE.evaluate_pipeline('GSE27887', 0.9)
# PE.evaluate_pipeline('GSE30999', 0.9)
# PE.evaluate_pipeline('GSE32924', 0.9)
# PE.evaluate_pipeline('GSE34248', 0.9)
# PE.evaluate_pipeline('GSE41662', 0.9)
# PE.evaluate_pipeline('GSE78097', 0.9)

data_names = ['GSE13355','GSE30999','GSE34248','GSE41662','GSE78097','GSE14905','GSE32924','GSE27887']


def calculate_scores():
    scores_10 = np.zeros((len(data_names), len(data_names),10))
    for i in range(len(data_names)):
        for j in range(len(data_names)):
            for k in range(10):
                print('Run %i' % k)
                scores_10[i, j, k] = CPE.cross_evaluate_pipeline(data_names[i], data_names[j], 0.75)

    scores = np.average(scores_10, 2)
    np.save('scores_matrix', scores)

order = np.argsort(np.array(data_names))

# calculate_scores()
scores = np.load('scores_matrix.npy')
scores = scores[:, order]

plot_figure = ['-.o', 's-.', 'p-.', '*-.', 'h-.', '<-.', 'D-.', '>-.']

legend_names = ['RandomForest', 'LinearSVC', 'DecisionTree', 'KNearestNeighbor', 'GradientBoosting','RandomForest', 'LinearSVC', 'RandomForest']

for i in range(scores.shape[0]):
    plt.plot(np.array(data_names)[order], scores[:, i], 'o-.', ms=15-2*i)
ax=plt.gca()
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.legend(legend_names)
plt.show()