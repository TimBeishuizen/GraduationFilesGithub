import csv
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import math

data_names = ['HeartAttack', 'Hepatitis', 'Cirrhosis', 'Cervical']
#data_names = ['Cirrhosis']

features = []
missing_percentage = []

rows = []

with open('combined_p_val.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        rows.append(row)

[features.append(feat) for feat in rows[2]]
[missing_percentage.append(float(perc)) for perc in rows[1]]
mean_p = np.asarray(rows[3:14]).swapaxes(0, 1).astype(float)
var_p = np.asarray(rows[14:25]).swapaxes(0, 1).astype(float)

for i in reversed(range(len(features))):
    cat = True
    for j in range(11):
        if not mean_p[i, j] in [1.0, 0.0]:
            cat = False
            break

    if missing_percentage[i] == 0 or cat:
        del features[i]
        del missing_percentage[i]
        mean_p = np.delete(mean_p, i, axis=0)
        var_p = np.delete(var_p, i, axis=0)

All_names = ['CCA', 'WCA', 'Mean', 'Hot deck', 'Regression', 'kNN, k=1', 'kNN, k=3', 'kNN, k=5',
             'MICE s=1', 'MICE s=3', 'MICE s=5']

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

fig, axes = plt.subplots(4, 3, sharex='all', sharey='all')

fig.text(0.5, 0.04, 'Missing values (%)', ha='center', va='center')
fig.text(0.06, 0.5, 'p-value mean distribution', ha='center', va='center', rotation='vertical')
fig.suptitle('Probability of the mean originating from both the old and new distribution'
             
             '\ndata points, linear regression line and R2 for validation of regression line'
             '\nafter removing features with more than 15% missing values,')


for j in range(11):
    fit = np.polyfit(mp, mean_p[order, j], fit_degree)
    axes[math.floor(j / 3), j % 3].scatter(mp * 100, mean_p[order, j])
    fit_output = poly_fit(x, fit)
    axes[math.floor(j/3), j%3].plot(x * 100, fit_output)

    fit_data = poly_fit(mp, fit)

    mean_y = np.mean(mean_p[order, j])
    SStot = sum([(fit_y - mean_y)**2 for fit_y in mean_p[order, j]])
    SSreg = sum([(reg_y - mean_y)**2 for reg_y in fit_output])
    SSres = sum([(mean_p[order, j][i] - fit_data[i])**2 for i in range(len(fit_data))])
    # R2 = 1 - (SSres / SStot)
    R2 = (SStot - SSres) / SStot

    axes[math.floor(j / 3), j % 3].set_title('%s, R2 = %.2f' % (All_names[j], R2))



    #axes[math.floor(j / 3), j % 3].set_xlabel('Missing values (%)')
    #axes[math.floor(j / 3), j % 3].set_ylabel('p-value mean distribution')
    #plt.legend(['regression line', 'data points'], loc='center left', bbox_to_anchor=(0.95, 0.5))

fig.show()
plt.show()

fig, axes = plt.subplots(4, 3, sharex='all', sharey='all')

fig.text(0.5, 0.04, 'Missing values (%)', ha='center', va='center')
fig.text(0.06, 0.5, 'p-value variance distribution', ha='center', va='center', rotation='vertical')
fig.suptitle('Probability of the variance originating from both the old and new distribution'
             
             '\ndata points, linear regression line and R2 for validation of regression line'
             '\nafter removing features with more than 15% missing values,')


for j in range(11):
    fit = np.polyfit(mp, var_p[order, j], fit_degree)
    axes[math.floor(j / 3), j % 3].scatter(mp * 100, var_p[order, j])
    fit_output = poly_fit(x, fit)
    axes[math.floor(j/3), j%3].plot(x * 100, fit_output)

    fit_data = poly_fit(mp, fit)

    mean_y = np.mean(var_p[order, j])
    SStot = sum([(fit_y - mean_y) ** 2 for fit_y in var_p[order, j]])
    SSreg = sum([(reg_y - mean_y) ** 2 for reg_y in fit_output])
    SSres = sum([(var_p[order, j][i] - fit_data[i]) ** 2 for i in range(len(fit_data))])
    R2 = 1 - (SSres / SStot)

    axes[math.floor(j / 3), j % 3].set_title('%s, R2 = %.2f' % (All_names[j], R2))
    # axes[math.floor(j / 3), j % 3].set_xlabel('Missing values (%)')
    # axes[math.floor(j / 3), j % 3].set_ylabel('p-value mean distribution')
    # plt.legend(['regression line', 'data points'], loc='center left', bbox_to_anchor=(0.95, 0.5))

fig.show()
plt.show()