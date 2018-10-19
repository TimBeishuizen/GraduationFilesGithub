import numpy as np
import scipy.stats as sp


def extract_significant_genes(sample_values, skin_types, gene_ids, target_groups=list([1, 2]), relation_groups='paired',
                              threshold=0.05, sorting=False, return_values='groups'):
    """ Extracts the genes that show a significant difference between the two data sets

    :param sample_values: The values of the gene expression for all samples
    :param skin_types: The values of the different skin types
    :param gene_ids: The ids of the genes
    :param target_groups: The two groups that should be tested for significance
    :param relation_groups: The relation between the two groups, can be:
            'paired' (default), 'equal variance' or 'unequal variance'
    :param threshold: The threshold of the p-value cut-off, default 0.05
    :param sorting: Whether the parameters need to be sorted or not, default false
    :param return_values: What sample values need to be returned. If 'groups'(default): sample values will be in groups.
        If 'all': sample values and skin_types will be returned
    :return: A list with only the significant genes and sample values
    """

    groups, remainder_skin_types = extract_target_groups(sample_values, skin_types, target_groups, remainder_group=True)
    group_remainder = np.array(groups[-1])
    group_1 = groups[0]
    group_2 = groups[1]

    # Do the t test for values
    if relation_groups == 'paired':
        t_value, p_value = sp.ttest_rel(group_1, group_2, axis=0, nan_policy='raise')
    elif relation_groups == 'equal variance':
        t_value, p_value = sp.ttest_ind(group_1, group_2, axis=0, nan_policy='raise')
    elif relation_groups == 'unequal variance':
        t_value, p_value = sp.ttest_ind(group_1, group_2, axis=0, equal_var=False, nan_policy='raise')
    else:
        raise NotImplementedError("%s is not implemented as a significance testing type" % relation_groups)

    if sorting:
        # Order everything
        order = np.argsort(p_value)

        p_value = p_value[order]
        group_1 = [value[order] for value in group_1]
        group_2 = [value[order] for value in group_2]
        gene_ids = [gene_ids[i] for i in order.tolist()]

    # Remove the values not significantly different according to the t-test
    sign_group_values = []
    sign_gene_ids = []

    np_group_1 = np.array(group_1)
    np_group_2 = np.array(group_2)

    for i in range(len(p_value)):
        if p_value[i] < threshold:
            sign_gene_ids.append(gene_ids[i])
            if group_remainder != []:
                sign_group_values.append(np.concatenate((np_group_1[:, i], np_group_2[:, i], group_remainder[:, i])).tolist())
            else:
                sign_group_values.append(
                    np.concatenate((np_group_1[:, i], np_group_2[:, i])).tolist())

        else:
            if sorting:
                break

    skin_type_values = [target_groups[0]] * np_group_1.shape[0] + [target_groups[1]] * np_group_2.shape[0] + remainder_skin_types

    print('%i show a significant difference between group %i and group %i' % (len(sign_gene_ids), target_groups[0],
                                                                              target_groups[1]))

    return sign_group_values, skin_type_values, sign_gene_ids


def extract_target_groups(sample_values, skin_types, target_groups, remainder_group = False):
    """ Extracts the target groups from the data sets
    
    :param sample_values: The values of the samples 
    :param skin_types: The skin types of the samples
    :param target_groups: The target groups of the significance testing
    :param remainder_group: Whether all other values should be added as remainder
    :return: The two target groups
    """
    
    groups = []
    
    for i in range(len(target_groups)):
        groups.append([])
        for j in range(sample_values.shape[0]):
            if target_groups[i] == skin_types[j]:
                groups[i].append(sample_values[j])

    # If remainder needs to be kept as well
    if remainder_group:
        groups.append([])
        remainder_skin_types = []
        for j in range(sample_values.shape[0]):
            if skin_types[j] not in target_groups:
                groups[-1].append(sample_values[j])
                remainder_skin_types.append(skin_types[j])

        return groups, remainder_skin_types

    return groups