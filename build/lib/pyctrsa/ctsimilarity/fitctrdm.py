# -*- coding: utf-8

"""
@File       :   fitctrdm.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating Cross-Temporal Similarities between CTRDMs and a Coding Model RDM '

import numpy as np
from pyctrsa.util.progressbar import show_progressbar
from pyctrsa.similarity import spearmanrp, pearsonrp, kendallrp, cosinesimilarity, euclideandistance


' a function for calculating Cross-Temporal Similarities between CTRDMs and a Coding Model RDM '

def ctsimilarities_cal(CTRDMs, Model_RDM, method='spearman'):

    """
    Calculate the Cross-Temporal Similarities between CTRDMs and a Coding Model RDM

    Parameters
    ----------
    CTRDMs : array
        The Cross-Temporal Representational Dissimilarity Matrices.
        The shape could be [n_ts, n_ts, n_conditions, n_conditions] or [n_subs, n_ts, n_ts, n_conditions, n_conditions]
        or [n_channels, n_ts, n_ts, n_conditions, n_conditionss] or [n_subs, n_channels, n_ts, n_ts, n_conditions,
        n_conditions]. n_ts, n_conditions, n_subs, n_channels represent the number of time-points, the number of
        conditions, the number of subjects and the number of channels, respectively.
    Model_RDM : array [n_conditions, n_conditions].
        The Coding Model RDM.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.

    Returns
    -------
    CTSimilarities : array
        Cross-temporal similarities.
        If method='spearman' or 'pearson' or 'kendall':
            If the shape of CTRDMs is [n_ts, n_ts, n_conditions, n_conditions], the shape of CTSimilarities will be
            [n_ts, n_ts, 2].
            If the shape of CTRDMs is [n_subs, n_ts, n_ts, n_conditions, n_conditions], the shape of CTSimilarities will
            be [n_subs, n_ts, n_ts, 2].
            If the shape of CTRDMs is [n_channels, n_ts, n_ts, n_conditions, n_conditions], the shape of CTSimilarities
            will be [n_channels, n_ts, n_ts, 2].
            If the shape of CTRDMs is [n_subs, n_channels, n_ts, n_ts, n_conditions, n_conditions], the shape of
            CTSimilarities will be [n_subs, n_channels, n_ts, n_ts, 2].
        If method='similarity' or 'distance':
            If the shape of CTRDMs is [n_ts, n_ts, n_conditions, n_conditions], the shape of CTSimilarities will be
            [n_ts, n_ts].
            If the shape of CTRDMs is [n_subs, n_ts, n_ts, n_conditions, n_conditions], the shape of CTSimilarities will
            be [n_subs, n_ts, n_ts].
            If the shape of CTRDMs is [n_channels, n_ts, n_ts, n_conditions, n_conditions], the shape of CTSimilarities
            will be [n_channels, n_ts, n_ts].
            If the shape of CTRDMs is [n_subs, n_channels, n_ts, n_ts, n_conditions, n_conditions], the shape of
            CTSimilarities will be [n_subs, n_channels, n_ts, n_ts].

    Notes
    -----
    Users can calculate CTRDMs by pyctrsa.ctrdm.single_cal module and pyctrsa.ctrdm.nulti_cal module
    (zitonglu1996.github.io/pyctrsa/)
    """

    n = len(np.shape(CTRDMs))

    if n == 4:

        n_ts, n_cons = np.shape(CTRDMs)[1:3]

        CTSimilarities = np.zeros([n_ts, n_ts, 2], dtype=np.float)

        total = n_ts * n_ts

        for t1 in range(n_ts):
            for t2 in range(n_ts):

                percent = (t1 * n_ts + t2) / total * 100
                show_progressbar("Calculating", percent)

                if method == 'spearman':
                    CTSimilarities[t1, t2] = spearmanrp.spearmanrp_cal(CTRDMs[t1, t2], Model_RDM)
                if method == 'pearson':
                    CTSimilarities[t1, t2] = pearsonrp.pearsonrp_cal(CTRDMs[t1, t2], Model_RDM)
                if method == 'kendall':
                    CTSimilarities[t1, t2] = kendallrp.kendallrp_cal(CTRDMs[t1, t2], Model_RDM)
                if method == 'similarity':
                    CTSimilarities[t1, t2, 0] = cosinesimilarity.cosinesimilarity_cal(CTRDMs[t1, t2], Model_RDM)
                if method == 'distance':
                    CTSimilarities[t1, t2, 0] = euclideandistance.euclideandistance_cal(CTRDMs[t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':

            return CTSimilarities

        if method == 'similarity' or method == 'distance':

            return CTSimilarities[:, :, 0]

    if n == 5:

        n1 = np.shape(CTRDMs)[0]
        n_ts, n_cons = np.shape(CTRDMs)[2:4]

        CTSimilarities = np.zeros([n1, n_ts, n_ts, 2], dtype=np.float)

        total = n1 * n_ts * n_ts

        for i in range(n1):
            for t1 in range(n_ts):
                for t2 in range(n_ts):

                    percent = (i * n_ts * n_ts + t1 * n_ts + t2) / total * 100
                    show_progressbar("Calculating", percent)

                    if method == 'spearman':
                        CTSimilarities[i, t1, t2] = spearmanrp.spearmanrp_cal(CTRDMs[i, t1, t2], Model_RDM)
                        #print(CTSimilarities[i, t1, t2])
                    if method == 'pearson':
                        CTSimilarities[i, t1, t2] = pearsonrp.pearsonrp_cal(CTRDMs[i, t1, t2], Model_RDM)
                    if method == 'kendall':
                        CTSimilarities[i, t1, t2] = kendallrp.kendallrp_cal(CTRDMs[i, t1, t2], Model_RDM)
                    if method == 'similarity':
                        CTSimilarities[i, t1, t2, 0] = cosinesimilarity.cosinesimilarity_cal(CTRDMs[i, t1, t2], Model_RDM)
                    if method == 'distance':
                        CTSimilarities[i, t1, t2, 0] = euclideandistance.euclideandistance_cal(CTRDMs[i, t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, 0]

    if n == 6:

        n1, n2 = np.shape(CTRDMs)[:2]
        n_ts, n_cons = np.shape(CTRDMs)[3:5]

        CTSimilarities = np.zeros([n1, n2, n_ts, n_ts, 2], dtype=np.float)

        total = n1 * n2 * n_ts * n_ts

        for i in range(n1):
            for j in range(n2):
                for t1 in range(n_ts):
                    for t2 in range(n_ts):

                        percent = (i * n2 * n_ts * n_ts + j * n_ts * n_ts + t1 * n_ts + t2) / total * 100
                        show_progressbar("Calculating", percent)

                        if method == 'spearman':
                            CTSimilarities[i, j, t1, t2] = spearmanrp.spearmanrp_cal(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'pearson':
                            CTSimilarities[i, j, t1, t2] = pearsonrp.pearsonrp_cal(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'kendall':
                            CTSimilarities[i, j, t1, t2] = kendallrp.kendallrp_cal(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'similarity':
                            CTSimilarities[i, j, t1, t2, 0] = cosinesimilarity.cosinesimilarity_cal(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'distance':
                            CTSimilarities[i, j, t1, t2, 0] = euclideandistance.euclideandistance_cal(CTRDMs[i, j, t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, :, 0]

# test codes
#a = np.random.rand(100, 100, 6, 6)
#c = np.random.rand(6, 6)
#b = ctsimilarities_cal(a, c)
#print(b)