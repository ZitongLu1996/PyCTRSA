# -*- coding: utf-8

"""
@File       :   normaldatabased.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating Cross-Temporal Similarities between neural data under two conditions '

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau


' a function for calculating Cross-Temporal Similarities between neural data under two conditions '

def ctsimilarities_cal(data1, data2, sub_opt=1, chl_opt=1, time_win=10, time_step=5, method='spearman'):

    """
    Calculate the Cross-Temporal Similarities between neural data under two conditions

    Parameters
    ----------
    data1 : array
        EEG/MEG data from a time-window under condition1.
        The shape of data should be [n_subs, n_channels, n_ts]. n_subs, n_channels, n_ts represent the number of
        conditions, the number of subjects, the number of channels and the number of time-points respectively.
    data2 : array
        EEG/MEG data from a time-window under condition2.
        The shape of data should be [n_subs, n_channels, n_ts]. n_subs, n_channels, n_ts represent the number of
        conditions, the number of subjects, the number of channels and the number of time-points respectively.
    sub_opt : int 0 or 1. Default is 1.
        Caculate the CTRDMs for each subject or not.
        If sub_opt=1, return the CTRDMs for each subjects.
        If sub_opt=0, return the avg CTRDMs among all subjects.
    chl_opt : int 0 or 1. Default is 1.
        Caculate the CTRDMs for each channel or not.
        If chl_opt=1, calculate the CTRDMs for each channel.
        If chl_opt=0, calculate the CTRDMs after averaging the channels.
    time_win : int. Default is 10.
        Set a time-window for calculating the CTRDM for different time-points.
        If time_win=10, that means each calculation process based on 10 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.
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
            If sub_opt=1 and chl_opt=1, the shape of CTSimilarities will be [n_subs, n_channels,
            int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1, 2]
            If sub_opt=1 and chl_opt=0, the shape of CTSimilarities will be [n_subs, int((n_ts-time_win)/time_step)+1,
            int((n_ts-time_win)/time_step)+1, 2]
            If sub_opt=0 and chl_opt=1, the shape of CTSimilarities will be [n_channels, int((n_ts-time_win)/time_step)
            +1, int((n_ts-time_win)/time_step)+1, 2]
            If sub_opt=0 and chl_opt=0, the shape of CTSimilarities will be [int((n_ts-time_win)/time_step)+1,
            int((n_ts-time_win)/time_step)+1, 2]
        If method='similarity' or 'distance':
            If sub_opt=1 and chl_opt=1, the shape of CTSimilarities will be [n_subs, n_channels,
            int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1]
            If sub_opt=1 and chl_opt=0, the shape of CTSimilarities will be [n_subs, int((n_ts-time_win)/time_step)+1,
            int((n_ts-time_win)/time_step)+1]
            If sub_opt=0 and chl_opt=1, the shape of CTSimilarities will be [n_channels, int((n_ts-time_win)/time_step)
            +1, int((n_ts-time_win)/time_step)+1]
            If sub_opt=0 and chl_opt=0, the shape of CTSimilarities will be [int((n_ts-time_win)/time_step)+1,
            int((n_ts-time_win)/time_step)+1]
    """

    n_subs, n_chls, n_ts = np.shape(data1)

    nts = int((n_ts - time_win) / time_step) + 1

    # chl_opt=0
    if chl_opt == 0:

        newdata1 = np.zeros([n_subs, nts, n_chls, time_win], dtype=np.float)
        newdata2 = np.zeros([n_subs, nts, n_chls, time_win], dtype=np.float)

        for sub in range(n_subs):
            for t in range(nts):
                for chl in range(n_chls):
                    newdata1[sub, t, chl] = data1[sub, chl, t*time_step:t*time_step+time_win]
                    newdata2[sub, t, chl] = data2[sub, chl, t*time_step:t*time_step+time_win]

        newdata1 = np.reshape(newdata1, [n_subs, nts, n_chls*time_win])
        newdata2 = np.reshape(newdata2, [n_subs, nts, n_chls*time_win])

        CTSimilarities = np.zeros([n_subs, nts, nts, 2], dtype=np.float)

        for sub in range(n_subs):
            for t1 in range(nts):
                for t2 in range(nts):

                    if method == 'spearman':
                        CTSimilarities[sub, t1, t2] = spearmanr(newdata1[sub, t1], newdata2[sub, t2])
                    if method == 'pearson':
                        CTSimilarities[sub, t1, t2] = pearsonr(newdata1[sub, t1], newdata2[sub, t2])
                    if method == 'kendall':
                        CTSimilarities[sub, t1, t2] = kendalltau(newdata1[sub, t1], newdata2[sub, t2])
                    if method == 'similarity':
                        V1 = np.mat(newdata1[sub, t1])
                        V2 = np.mat(newdata2[sub, t2])
                        num = float(V1 * V2.T)
                        denom = np.linalg.norm(V1) * np.linalg.norm(V2)
                        cos = num / denom
                        CTSimilarities[sub, t1, t2, 0] = 0.5 + 0.5 * cos
                    if method == 'distance':
                        CTSimilarities[sub, t1, t2, 0] = np.linalg.norm(newdata1[sub, t1] - newdata2[sub, t2])

        if sub_opt == 0:

            CTSimilarities = np.average(CTSimilarities, axis=0)

            if method == 'spearman' or method == 'pearson' or method == 'kendall':
                return CTSimilarities

            if method == 'similarity' or method == 'distance':
                return CTSimilarities[:, :, 0]

        if sub_opt == 1:

            if method == 'spearman' or method == 'pearson' or method == 'kendall':
                return CTSimilarities

            if method == 'similarity' or method == 'distance':
                return CTSimilarities[:, :, :, 0]

    if chl_opt == 1:

        newdata1 = np.zeros([n_subs, n_chls, nts, time_win], dtype=np.float)
        newdata2 = np.zeros([n_subs, n_chls, nts, time_win], dtype=np.float)

        for sub in range(n_subs):
            for chl in range(n_chls):
                for t in range(nts):
                    newdata1[sub, chl, t] = data1[sub, chl, t * time_step:t * time_step + time_win]
                    newdata2[sub, chl, t] = data2[sub, chl, t * time_step:t * time_step + time_win]

        CTSimilarities = np.zeros([n_subs, n_chls, nts, nts, 2], dtype=np.float)

        for sub in range(n_subs):
            for chl in range(n_chls):
                for t1 in range(nts):
                    for t2 in range(nts):

                        if method == 'spearman':
                            CTSimilarities[sub, chl, t1, t2] = spearmanr(newdata1[sub, chl, t1], newdata2[sub, chl, t2])
                        if method == 'pearson':
                            CTSimilarities[sub, chl, t1, t2] = pearsonr(newdata1[sub, chl, t1], newdata2[sub, chl, t2])
                        if method == 'kendall':
                            CTSimilarities[sub, chl, t1, t2] = kendalltau(newdata1[sub, chl, t1], newdata2[sub, chl, t2])
                        if method == 'similarity':
                            V1 = np.mat(newdata1[sub, chl, t1])
                            V2 = np.mat(newdata2[sub, chl, t2])
                            num = float(V1 * V2.T)
                            denom = np.linalg.norm(V1) * np.linalg.norm(V2)
                            cos = num / denom
                            CTSimilarities[sub, t1, t2, 0] = 0.5 + 0.5 * cos
                        if method == 'distance':
                            CTSimilarities[sub, t1, t2, 0] = np.linalg.norm(newdata1[sub, chl, t1] - newdata2[sub, chl, t2])

        if sub_opt == 0:

            CTSimilarities = np.average(CTSimilarities, axis=0)

            if method == 'spearman' or method == 'pearson' or method == 'kendall':
                return CTSimilarities

            if method == 'similarity' or method == 'distance':
                return CTSimilarities[:, :, :, 0]

        if sub_opt == 1:

            if method == 'spearman' or method == 'pearson' or method == 'kendall':
                return CTSimilarities

            if method == 'similarity' or method == 'distance':
                return CTSimilarities[:, :, :, :, 0]



