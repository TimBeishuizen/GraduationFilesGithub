import csv
import numpy as np
import matplotlib.pyplot as plt
import math

data_names = ['HeartAttack', 'Hepatitis', 'Cirrhosis', 'Cervical']
#data_names = ['Cirrhosis']

features = []
missing_percentage = []

mean_p = np.zeros((0, 7))
var_p = np.zeros((0, 7))

for data_name in data_names:

    rows = []

    with open(data_name + '_p_val.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            rows.append(row)

    [features.append(feat) for feat in rows[0]]
    [missing_percentage.append(float(perc)) for perc in rows[1]]
    mean_p = np.append(mean_p, np.transpose(np.asarray(rows[2:9])).astype(float), axis=0)
    var_p = np.append(var_p, np.transpose(np.asarray(rows[9:16])).astype(float), axis=0)


mean_p.astype(float)
var_p.astype(float)

for i in reversed(range(len(features))):
    cat = True
    for j in range(7):
        if not mean_p[i, j] in [1.0, 0.0]:
            cat = False
            break

    if missing_percentage[i] == 0 or cat:
        del features[i]
        del missing_percentage[i]
        mean_p = np.delete(mean_p, i, axis=0)
        var_p = np.delete(var_p, i, axis=0)

All_names = ['CCA', 'WCA', 'Mean', 'Hot deck', 'kNN, k=3', 'Regression', 'MICE']

order = np.argsort(missing_percentage)[:-5]

mp = np.asarray(missing_percentage)[order]

x = np.arange(0.0, 0.15, step=0.01)
fit_degree = 1

def poly_fit(x, args):
    """

    :param x:
    :param deg:
    :param args:
    :return:
    """

    y = args[-1]

    deg = len(args) - 1

    for i in range(deg):
        y += x ** (deg - i) * args[i]

    return y

fig, axes = plt.subplots(3, 3, sharex='all', sharey='all')

fig.text(0.5, 0.04, 'Missing values (%)', ha='center', va='center')
fig.text(0.06, 0.5, 'p-value mean distribution', ha='center', va='center', rotation='vertical')
fig.suptitle('Probability of the mean originating from both the old and new distribution,'
                                             '\nboth data points and linear regression line')


for j in range(7):
    fit = np.polyfit(mp, mean_p[order, j], fit_degree)
    axes[math.floor(j / 3), j % 3].scatter(mp * 100, mean_p[order, j])
    fit_output = poly_fit(x, fit)
    axes[math.floor(j/3), j%3].plot(x * 100, fit_output)

    axes[math.floor(j / 3), j % 3].set_title('%s' % All_names[j])
    #axes[math.floor(j / 3), j % 3].set_xlabel('Missing values (%)')
    #axes[math.floor(j / 3), j % 3].set_ylabel('p-value mean distribution')
    #plt.legend(['regression line', 'data points'], loc='center left', bbox_to_anchor=(0.95, 0.5))

fig.show()
plt.show()

fig, axes = plt.subplots(3, 3, sharex='all', sharey='all')

fig.text(0.5, 0.04, 'Missing values (%)', ha='center', va='center')
fig.text(0.06, 0.5, 'p-value variance distribution', ha='center', va='center', rotation='vertical')
fig.suptitle('Probability of the variance originating from both the old and new distribution,'
                                             '\nboth data points and linear regression line')


for j in range(7):
    fit = np.polyfit(mp, var_p[order, j], fit_degree)
    axes[math.floor(j / 3), j % 3].scatter(mp * 100, var_p[order, j])
    fit_output = poly_fit(x, fit)
    axes[math.floor(j/3), j%3].plot(x * 100, fit_output)

    axes[math.floor(j / 3), j % 3].set_title('%s' % All_names[j])
    #axes[math.floor(j / 3), j % 3].set_xlabel('Missing values (%)')
    #axes[math.floor(j / 3), j % 3].set_ylabel('p-value mean distribution')
    #plt.legend(['regression line', 'data points'], loc='center left', bbox_to_anchor=(0.95, 0.5))

fig.show()
plt.show()