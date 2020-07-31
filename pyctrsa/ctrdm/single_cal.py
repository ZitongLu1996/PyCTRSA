# -*- coding: utf-8

"""
@File       :   ctrdm_cal.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating Cross-Temporal RDMs for a single channel & a single subject '

import numpy as np
from pyctrsa.util.progressbar import show_progressbar
from scipy.stats import pearsonr

np.seterr(divide='ignore', invalid='ignore')


' A function to calculate Cross-Temporal RDMs for a single channel & a single subject '

def ctrdm_cal(data, time_win=10, time_step=5):

    """
    a function to calculate CTRDMs for a single channel & a single subject

    Parameters
    ----------
    data : array
        EEG/MEG data from a time-window.
        The shape of data should be [n_conditions, n_ts]. n_conditions, n_ts represent the number of conditions and the
        number of time-points respectively.
    time_win : int. Default is 10.
        Set a time-window for calculating the CTRDM for different time-points.
        If time_win=10, that means each calculation process based on 10 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.

    Returns
    -------
    CTRDMs : array [int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1, n_conditions, n_conditions]
        Cross-Temporal RDMs.
    """

    n_cons, n_ts = np.shape(data)

    nts = int((n_ts - time_win) / time_step) + 1

    data_for_cal = np.zeros([n_cons, nts, time_win], dtype=np.float)

    for con in range(n_cons):
        for t in range(nts):
            data_for_cal[con, t] = data[con, t*time_step:t*time_step+time_win]

    ctrdms = np.zeros([nts, nts, n_cons, n_cons], dtype=np.float)

    total = nts*nts

    for t1 in range(nts):
        for t2 in range(nts):

            percent = (t1*nts + t2)/total*100
            show_progressbar("Calculating", percent)

            for con1 in range(n_cons):
                for con2 in range(n_cons):

                    if con1 != con2:
                        r = pearsonr(data_for_cal[con1, t1], data_for_cal[con2, t2])[0]
                        ctrdms[t1, t2, con1, con2] = 1-r
                    if con1 == con2:
                        ctrdms[t1, t2, con1, con2] = 0

    return ctrdms

# test
#data = np.random.rand(5, 80)
#print(ctrdm_cal(data)[1, 2])

#from pyctrsa.plotting import ctrdm
#ctrdm.ctrdm_plot(ctrdm_cal(data)[1, 2])
