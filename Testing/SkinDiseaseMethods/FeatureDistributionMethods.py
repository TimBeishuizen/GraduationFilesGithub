import numpy as np

import scipy.stats as sp

from matplotlib import pyplot as plt


def compute_distribution_values(X, missing_values=None):
    """

    :param X:
    :return:
    """

    X = np.asarray(X)

    feature_specs = []

    for i in range(X.shape[1]):
        X_f = X[:, i]
        X_f = np.delete(X_f, np.where(X_f == missing_values), axis=0)

        try:
            X_f = X_f.astype(float)

            # Get mean and variance
            mean_f = np.mean(X_f)
            var_f = np.var(X_f)
            skewness_f = sp.skew(X_f)
            kurtosis_f = sp.kurtosis(X_f)

            feature_specs.append([mean_f, var_f, skewness_f, kurtosis_f])

        except ValueError:
            # Get mode from categories
            mode_f, _ = sp.mode(X_f)

            # Sum of squares fractions
            categories = np.unique(X_f)
            avg_size = X_f.shape[0] / categories.shape[0]
            sum_squared = 0

            # Find the variance in category count
            for category in categories:
                count = np.count_nonzero(X_f == category)

                sum_squared = abs(count - avg_size) #** 2

            feature_specs.append([np.asscalar(mode_f), sum_squared])

    return feature_specs


def show_distribution_histograms(final_data, dist_names, missing_values=None):
    """

    :param final_data:
    :param dist_names:
    :param missing_values:
    :return:
    """

    for i in range(final_data[0].shape[1]):

        hist_data = []
        hist = True
        for j in range(len(final_data)):

            final_array = final_data[j][:, i]
            data = np.delete(final_array, np.argwhere(final_array == missing_values))

            try:
                hist_data.append(data.astype(float))
                hist = True
            except ValueError:
                #hist = False
                hist_data.append(data)
                #plt.hist(data, bins=20, label=dist_names[j], alpha=0.3)

        plt.hist(hist_data, bins=20, label=dist_names, alpha=0.5, density=True)

        plt.title("The distributions of feature %i" %i)
        plt.legend()
        plt.show()


def show_specs_distributions(feature_specs, dist_names, off_set, missing_percentage):
    """

    :param feature_specs:
    :param dist_names:
    :param off_set:
    :param missing_percentage:
    :return:
    """

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i in range(len(feature_specs[0])):

        mean = []
        var = []
        skew = []
        kurt = []

        for j in range(len(feature_specs)):
            mean.append(feature_specs[j][i][0])
            var.append(feature_specs[j][i][1])
            if len(feature_specs[j][i]) > 2:
                skew.append(feature_specs[j][i][2])
                kurt.append(feature_specs[j][i][3])

        if skew == []:
            fig, ax = plt.subplots(1, 2)
        else:
            fig, ax = plt.subplots(1, 4)

            ax[2].bar(dist_names, height=skew, color=color[:len(feature_specs)], alpha=0.7)
            ax[2].plot([-1, 7], [off_set[i][2], off_set[i][2]])
            ax[2].set_title("Skewness")

            for tick in ax[2].get_xticklabels():
                tick.set_rotation(45)

            ax[3].bar(dist_names, height=kurt, color=color[:len(feature_specs)], alpha=0.7)
            ax[3].plot([-1, 7], [off_set[i][3], off_set[i][3]])
            ax[3].set_title("Kurtosis")

            for tick in ax[3].get_xticklabels():
                tick.set_rotation(45)

        plt.suptitle("Feature %i with %f percent missing values" % (i, missing_percentage[i]*100))

        ax[0].bar(dist_names, height=mean, color=color[:len(feature_specs)], alpha=0.7)
        ax[0].plot([-1, 7], [off_set[i][0], off_set[i][0]])
        ax[0].set_title("Mean")

        for tick in ax[0].get_xticklabels():
            tick.set_rotation(45)

        ax[1].bar(dist_names, height=var, color=color[:len(feature_specs)], alpha=0.7)
        ax[1].plot([-1, 7], [off_set[i][1], off_set[i][1]])
        ax[1].set_title("Variance")

        for tick in ax[1].get_xticklabels():
            tick.set_rotation(45)

        plt.show()


def compute_distribution_similarities(X_obs, X_ref, missing_values=None):
    """

    :param X_1:
    :param X_2:
    :return:
    """

    p_values = []
    var_p_values = []

    for i in range(X_obs.shape[1]):
        X_f = X_ref[:, i]
        X_f = np.delete(X_f, np.where(X_f == missing_values), axis=0)

        try:
            X_f = X_f.astype(float)

            if np.unique(X_f).shape[0] <= 2:
                categories = np.unique(X_f)

                f_obs = [np.count_nonzero(X_obs[:, i] == c) / X_obs.shape[0] for c in categories]
                f_ref = [np.count_nonzero(X_f == c) / X_f.shape[0] for c in categories]

                p_value = (X_obs[:, i] == np.median(X_f))
                chi_sq, var_p_value = sp.chisquare(f_obs, f_ref)
            else:
                t_value, p_value = sp.ttest_ind(X_obs[:, i].astype(float), X_f, equal_var=False)
                W_value, var_p_value = sp.levene(X_obs[:, i].astype(float), X_f)

        except ValueError:
            categories = np.unique(X_f)
            f_obs = [np.count_nonzero(X_obs[:, i] == c) / X_obs.shape[0] for c in categories]
            f_ref = [np.count_nonzero(X_f == c) / X_f.shape[0] for c in categories]

            mode_obs, _ = sp.mode(X_obs[:, i])
            mode_ref, _ = sp.mode(X_f)

            p_value = (mode_obs[0] == mode_ref[0])
            chi_sq, var_p_value = sp.chisquare(f_obs, f_ref)

        p_values.append(float(p_value))
        var_p_values.append(var_p_value)

    return p_values, var_p_values


def plot_p_relation(p_val, dist_names, missing_percentage):
    """

    :param p_val:
    :param missing_percentage:
    :return:
    """

    for i in range(len(p_val)):
        plt.scatter(missing_percentage, p_val[i])
    plt.legend(dist_names)
    plt.xlabel("Missing percentage")
    plt.ylabel("P-value")
    plt.show()
