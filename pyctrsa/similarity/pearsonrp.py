# -*- coding: utf-8

"""
@File       :   pearsonrp.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating the similarity based on Pearson Correlation between two CTRDMs '

import numpy as np
from neurora.stuff import fisherz_rdm
from scipy.stats import pearsonr


' a function for calculating the similarity based on Pearson Correlation Coefficient between two CTRDMs '

def pearsonrp_cal(CTRDM1, CTRDM2):
    """
    Calculate the similarity based on Pearson Correlation Coefficient between two CTRDMs

    Parameters
    ----------
    CTRDM1 : array [n_conditions, n_conditions]
        The Cross-Temporal RDM 1.
    CTRDM2 : array [n_conditions, n_conditions]
        The Cross-Temporal RDM 2.

    Returns
    -------
    rp : float, float
        Pearson's correlation coefficient
        A r-value and a p-value.
    """

    # get number of conditions
    n_cons = np.shape(CTRDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = n_cons * (n_cons - 1)

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    # assignment
    nn = 0
    for i in range(n_cons):
        for j in range(n_cons):
            if i != j:
                v1[nn] = CTRDM1[i, j]
                v2[nn] = CTRDM2[i, j]
                nn = nn + 1

    # calculate the Pearson Correlation
    rp = np.array(pearsonr(v1, v2))

    return rp