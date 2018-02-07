import numpy as np
import scipy.stats as sp


def extract_significant_genes(sample_values, skin_types, gene_ids, threshold=0.05, sorting=False):
    """ Extracts the genes that show a significant difference between the two data sets

    :param sample_values: The values of the gene expression for all samples
    :param skin_types: The values of the different skin types
    :param gene_ids: The ids of the genes
    :param threshold: The threshold of the p-value cut-off, default 0.05
    :param sorting: Whether the parameters need to be sorted or not, default false
    :return: A list with only the significant genes and sample values
    """

    N_values = []
    NL_values = []
    L_values = []
    E_values = []

    for i in range(sample_values.shape[0]):
        if skin_types[i] == 0:
            N_values.append(sample_values[i])
        elif skin_types[i] == 1:
            NL_values.append(sample_values[i])
        elif skin_types[i] == 2:
            L_values.append(sample_values[i])
        elif skin_types[i] == 3:
            E_values.append(sample_values[i])
        else:
            raise NotImplementedError("An unknown skin-type was used")

    # Do the t test for values
    t_value, p_value = sp.ttest_rel(NL_values, L_values, axis=0, nan_policy='raise')

    if sorting == True:
        # Order everything
        order = np.argsort(p_value)

        p_value = p_value[order]
        NL_values = [value[order] for value in NL_values]
        L_values = [value[order] for value in L_values]
        gene_ids = [gene_ids[i] for i in order]

    # Remove the values not significantly different according to the t-test
    sign_NL_values = []
    sign_L_values = []
    sign_gene_ids = []

    np_NL = np.array(NL_values)
    np_L = np.array(L_values)

    for i in range(len(p_value)):
        if p_value[i] < threshold:
            sign_NL_values.append(np_NL[:, i].tolist())
            sign_L_values.append(np_L[:, i].tolist())
            sign_gene_ids.append(gene_ids[i])
        else:
            if sorting:
                break

    return sign_NL_values, sign_L_values, sign_gene_ids
