# -*- coding: utf-8

"""
@File       :   normalrdmbased.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating Cross-Temporal Similarities based on normal RDMs '

import numpy as np
from neurora.rdm_corr import rdm_correlation_spearman, rdm_correlation_pearson, rdm_correlation_kendall
from neurora.rdm_corr import rdm_similarity, rdm_distance


' a function for calculating Cross-Temporal Similarities based on normal RDMs '

def ctsimilarities_cal(RDMs, method='spearman', fisherz=True):

    """
    Calculate the Cross-Temporal Similarities based on normal RDMs

    Parameters
    ----------
    RDMs : array
        The Representational Dissimilarity Matrices in time series.
        The shape could be [n_ts, n_conditions, n_conditions] or [n_subs, n_ts, n_conditions, n_conditions] or
        [n_channels, n_ts, n_conditions, n_conditions] or [n_subs, n_channels, n_ts, n_conditions, n_conditions].
        n_ts, n_conditions, n_subs, n_channels represent the number of time-points, the number of conditions, the number
        of subjects and the number of channels, respectively.
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
        If the shape of RDMs is [n_ts, n_conditions, n_conditions] and method='spearman' or 'pearson' or 'kendall', the
        shape of CTSimilarities will be [n_ts, n_ts, 2].
        If the shape of RDMs is [n_subs, n_ts, n_conditions, n_conditions] and method='spearman' or 'pearson' or
        'kendall', the shape of CTSimilarities will be [n_subs, n_ts, n_ts, 2].
        If the shape of RDMs is [n_channels, n_ts, n_conditions, n_conditions] and method='spearman' or 'pearson' or
        'kendall', the shape of CTSimilarities will be [n_channels, n_ts, n_ts, 2].
        If the shape of RDMs is [n_subs, n_channels, n_ts, n_conditions, n_conditions] and method='spearman' or
        'pearson' or 'kendall', the shape of CTSimilarities will be [n_subs, n_channels, n_ts, n_ts, 2].
        If the shape of RDMs is [n_ts, n_conditions, n_conditions] and method='similarity' or 'distance', the shape of
        CTSimilarities will be [n_ts, n_ts].
        If the shape of RDMs is [n_subs, n_ts, n_conditions, n_conditions] and method='similarity' or 'distance', the
        shape of CTSimilarities will be [n_subs, n_ts, n_ts].
        If the shape of RDMs is [n_channels, n_ts, n_conditions, n_conditions] and method='similarity' or 'distance',
        the shape of CTSimilarities will be [n_channels, n_ts, n_ts].
        If the shape of RDMs is [n_subs, n_channels, n_ts, n_conditions, n_conditions] and method='similarity' or
        'distance', the shape of CTSimilarities will be [n_subs, n_channels, n_ts, n_ts].

    Notes
    -----
    Users can calculate RDMs by NeuroRA (zitonglu1996.github.io/neurora/)
    """

    n = len(np.shape(RDMs))

    if n == 3:

        n_ts, n_cons = np.shape(RDMs)[:2]

        CTSimilarities = np.zeros([n_ts, n_ts, 2], dtype=np.float)

        for t1 in range(n_ts):
            for t2 in range(n_ts):

                if method == 'spearman':
                    CTSimilarities[t1, t2] = rdm_correlation_spearman(RDMs[t1], RDMs[t2], fisherz=fisherz)
                if method == 'pearson':
                    CTSimilarities[t1, t2] = rdm_correlation_pearson(RDMs[t1], RDMs[t2], fisherz=fisherz)
                if method == 'kendall':
                    CTSimilarities[t1, t2] = rdm_correlation_kendall(RDMs[t1], RDMs[t2], fisherz=fisherz)
                if method == 'similarity':
                    CTSimilarities[t1, t2, 0] = rdm_similarity(RDMs[t1], RDMs[t2])
                if method == 'distance':
                    CTSimilarities[t1, t2, 0] = rdm_distance(RDMs[t1], RDMs[t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':

            return CTSimilarities

        if method == 'similarity' or method == 'distance':

            return CTSimilarities[:, :, 0]

    if n == 4:

        n1, n_ts, n_cons = np.shape(RDMs)[:3]

        CTSimilarities = np.zeros([n1, n_ts, n_ts, 2], dtype=np.float)

        for i in range(n1):
            for t1 in range(n_ts):
                for t2 in range(n_ts):

                    if method == 'spearman':
                        CTSimilarities[i, t1, t2] = rdm_correlation_spearman(RDMs[i, t1], RDMs[i, t2], fisherz=fisherz)
                    if method == 'pearson':
                        CTSimilarities[i, t1, t2] = rdm_correlation_pearson(RDMs[i, t1], RDMs[i, t2], fisherz=fisherz)
                    if method == 'kendall':
                        CTSimilarities[i, t1, t2] = rdm_correlation_kendall(RDMs[i, t1], RDMs[i, t2], fisherz=fisherz)
                    if method == 'similarity':
                        CTSimilarities[i, t1, t2, 0] = rdm_similarity(RDMs[i, t1], RDMs[i, t2])
                    if method == 'distance':
                        CTSimilarities[i, t1, t2, 0] = rdm_distance(RDMs[i, t1], RDMs[i, t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, 0]

    if n == 5:

        n1, n2, n_ts, n_cons = np.shape(RDMs)[:4]

        CTSimilarities = np.zeros([n1, n2, n_ts, n_ts, 2], dtype=np.float)

        for i in range(n1):
            for j in range(n2):
                for t1 in range(n_ts):
                    for t2 in range(n_ts):

                        if method == 'spearman':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_spearman(RDMs[i, j, t1], RDMs[i, j, t2],
                                                                                 fisherz=fisherz)
                        if method == 'pearson':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_pearson(RDMs[i, j, t1], RDMs[i, j, t2],
                                                                                fisherz=fisherz)
                        if method == 'kendall':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_kendall(RDMs[i, j, t1], RDMs[i, j, t2],
                                                                                fisherz=fisherz)
                        if method == 'similarity':
                            CTSimilarities[i, j, t1, t2, 0] = rdm_similarity(RDMs[i, j, t1], RDMs[i, j, t2])
                        if method == 'distance':
                            CTSimilarities[i, j, t1, t2, 0] = rdm_distance(RDMs[i, j, t1], RDMs[i, j, t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, :, 0]
