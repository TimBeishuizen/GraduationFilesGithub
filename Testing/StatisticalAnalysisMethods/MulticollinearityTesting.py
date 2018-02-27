import time
import numpy as np


def greedy_cluster_genes(gene_values, gene_ids, corr_threshold=0.7, test_number=None, info_print=True):
    """ Greedy clusters the genes into having one main gene that represent the others

    :param gene_values: All values of the genes for the different samples
    :param gene_ids: The ids for the genes in matrix
    :param corr_threshold: The threshold of the correlation coefficient to be regarded as correlated
    :param test_number: The number of genes that needs to be tested
    :param info_print: If additional information on the clusters should be printed, default TRUE
    :return: The uncorrelated representative genes and their ID's, as well as the gene values and the ids of the genes
            they are representing
    """

    uncorr_values = []
    corr_values = []
    uncorr_gene_ids = []
    corr_gene_ids = []
    uncorrelated = 0

    start_time = time.time()

    if test_number == None:
        test_number = gene_values.shape[0]

    # Check for multicollinearity
    print("Checking for multicollinearity...")
    for i in range(test_number):

        # After testing 5% of the genes, give an update
        if i % (int(test_number/20)) == 0:
            print("Currently at gene %i, %i uncorrelated genes found in %i seconds" %
                  (i, uncorrelated, (time.time() - start_time)))

        # Test whether the genes correlate too much with one of the other genes
        for j in range(len(uncorr_values)):
            pearson_correlation = np.corrcoef(gene_values[i, :], uncorr_values[j])[0, 1]

            # Add values of the new gene to the gene it correlates to
            if abs(pearson_correlation) > corr_threshold:
                corr_values[j].append(gene_values[i, :])
                corr_gene_ids[j].append(gene_ids[i])
                break

        # If no correlation was found add the gene to the uncorrelated genes
        else:
            uncorr_values.append(gene_values[i, :])
            corr_values.append([gene_values[i, :]])
            uncorr_gene_ids.append(gene_ids[i])
            corr_gene_ids.append([gene_ids[i]])
            uncorrelated += 1

    print("%i clusters are found" % len(corr_gene_ids))
    for i in range(len(corr_gene_ids)):
        correlation = np.min(abs(np.corrcoef(corr_values[i])))
        if len(corr_gene_ids[i]) > 10 and info_print:
            print("Cluster: %i, number of genes: %i, minimum correlation: %f" % (
            i + 1, len(corr_gene_ids[i]), correlation))

    uncorr_values = np.asarray(uncorr_values)

    return uncorr_values, uncorr_gene_ids, corr_values, corr_gene_ids
