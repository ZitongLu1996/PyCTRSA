# -*- coding: utf-8

"""
@File       :   fitctrdm.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating Cross-Temporal Similarities between CTRDMs and a Coding Model RDM '

import numpy as np
from pyctrsa.similarity import spearmanrp, pearsonrp, kendallrp, cosinesimilarity, euclideandistance


' a function for calculating Cross-Temporal Similarities between CTRDMs and a Coding Model RDM '

def ctsimilarities_cal(CTRDMs, Model_RDM, method='spearman', fisherz=True):

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
    fisherz : bool True or False. Default is False.
        Do the Fisher-Z transform of the CTRDMs or not.
        Only when method='spearman' or 'pearson' or 'kendall', it works.

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

    if n == 3:

        n_ts, n_cons = np.shape(CTRDMs)[1:3]

        CTSimilarities = np.zeros([n_ts, n_ts, 2], dtype=np.float)

        for t1 in range(n_ts):
            for t2 in range(n_ts):

                if method == 'spearman':
                    CTSimilarities[t1, t2] = spearmanrp(CTRDMs[t1, t2], Model_RDM, fisherz=fisherz)
                if method == 'pearson':
                    CTSimilarities[t1, t2] = pearsonrp(CTRDMs[t1, t2], Model_RDM, fisherz=fisherz)
                if method == 'kendall':
                    CTSimilarities[t1, t2] = kendallrp(CTRDMs[t1, t2], Model_RDM, fisherz=fisherz)
                if method == 'similarity':
                    CTSimilarities[t1, t2, 0] = cosinesimilarity(CTRDMs[t1, t2], Model_RDM)
                if method == 'distance':
                    CTSimilarities[t1, t2, 0] = euclideandistance(CTRDMs[t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':

            return CTSimilarities

        if method == 'similarity' or method == 'distance':

            return CTSimilarities[:, :, 0]

    if n == 4:

        n1, n_ts, n_cons = np.shape(CTRDMs)[2:4]

        CTSimilarities = np.zeros([n1, n_ts, n_ts, 2], dtype=np.float)

        for i in range(n1):
            for t1 in range(n_ts):
                for t2 in range(n_ts):

                    if method == 'spearman':
                        CTSimilarities[i, t1, t2] = spearmanrp(CTRDMs[i, t1, t2], Model_RDM, fisherz=fisherz)
                    if method == 'pearson':
                        CTSimilarities[i, t1, t2] = pearsonrp(CTRDMs[i, t1, t2], Model_RDM, fisherz=fisherz)
                    if method == 'kendall':
                        CTSimilarities[i, t1, t2] = kendallrp(CTRDMs[i, t1, t2], Model_RDM, fisherz=fisherz)
                    if method == 'similarity':
                        CTSimilarities[i, t1, t2, 0] = cosinesimilarity(CTRDMs[i, t1, t2], Model_RDM)
                    if method == 'distance':
                        CTSimilarities[i, t1, t2, 0] = euclideandistance(CTRDMs[i, t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, 0]

    if n == 5:

        n1, n2, n_ts, n_cons = np.shape(CTRDMs)[:4]

        CTSimilarities = np.zeros([n1, n2, n_ts, n_ts, 2], dtype=np.float)

        for i in range(n1):
            for j in range(n2):
                for t1 in range(n_ts):
                    for t2 in range(n_ts):

                        if method == 'spearman':
                            CTSimilarities[i, j, t1, t2] = spearmanrp(CTRDMs[i, j, t1, t2], Model_RDM, fisherz=fisherz)
                        if method == 'pearson':
                            CTSimilarities[i, j, t1, t2] = pearsonrp(CTRDMs[i, j, t1, t2], Model_RDM, fisherz=fisherz)
                        if method == 'kendall':
                            CTSimilarities[i, j, t1, t2] = kendallrp(CTRDMs[i, j, t1, t2], Model_RDM, fisherz=fisherz)
                        if method == 'similarity':
                            CTSimilarities[i, j, t1, t2, 0] = cosinesimilarity(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'distance':
                            CTSimilarities[i, j, t1, t2, 0] = euclideandistance(CTRDMs[i, j, t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, :, 0]
