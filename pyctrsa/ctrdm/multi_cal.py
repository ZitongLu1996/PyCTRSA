# -*- coding: utf-8

"""
@File       :   multi_cal.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating Cross-Temporal RDMs for multi-channels data '

import numpy as np
from pyctrsa.util.progressbar import show_progressbar
from pyctrsa.ctrdm import single_cal
from scipy.stats import pearsonr

np.seterr(divide='ignore', invalid='ignore')


' A function to calculate Cross-Temporal RDMs for multi-channels '

def ctrdms_cal(data, sub_opt=1, chl_opt=1, time_win=10, time_step=5):

    """
    a function to calculate CTRDMs for multi-channels

    Parameters
    ----------
    data : array
        EEG/MEG data from a time-window.
        The shape of data should be [n_conditions, n_subs, n_channels, n_ts]. n_conditions, n_subs, n_channels, n_ts
        represent the number of conditions, the number of subjects, the number of channels and the number of time-points
        respectively.
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

    Returns
    -------
    CTRDMs : array
        Cross-Temporal RDMs.
        if chl_opt=1, the shape of CTRDMs is [n_subs, n_chls, int((n_ts-time_win)/time_step)+1,
        int((n_ts-time_win)/time_step)+1, n_cons, n_cons]
        if chl_opt=0, the shape of CTRDMs is [n_subs, int((n_ts-time_win)/time_step)+1,
        int((n_ts-time_win)/time_step)+1, n_cons, n_cons]
    """

    n_cons, n_subs, n_chls, n_ts = np.shape(data)

    nts = int((n_ts - time_win) / time_step) + 1

    # chl_opt=0
    if chl_opt == 0:

        data_for_cal = np.zeros([n_cons, n_subs, nts, n_chls, time_win], dtype=np.float)

        for con in range(n_cons):
            for sub in range(n_subs):
                for t in range(nts):
                    for chl in range(n_chls):
                        data_for_cal[con, sub, t, chl] = data[con, sub, chl, t * time_step:t * time_step + time_win]

        data_for_cal = np.reshape(data_for_cal, [n_cons, n_subs, nts, n_chls*time_win])

        ctrdms = np.zeros([n_subs, nts, nts, n_cons, n_cons], dtype=np.float)

        total = n_subs * nts * nts

        for sub in range(n_subs):
            for t1 in range(nts):
                for t2 in range(nts):
                    percent = (sub * nts * nts + t1 * nts + t2) / total * 100
                    show_progressbar("Calculating", percent)

                    for con1 in range(n_cons):
                        for con2 in range(n_cons):

                            if con1 != con2:
                                r = pearsonr(data_for_cal[con1, sub, t1], data_for_cal[con2, sub, t2])[0]
                                ctrdms[sub, t1, t2, con1, con2] = 1 - r
                            if con1 == con2:
                                ctrdms[sub, t1, t2, con1, con2] = 0

        # chl_opt=0 & sub_opt=0
        if sub_opt == 0:

            return np.average(ctrdms, axis=0)

        # chl_opt=0 & sub_opt=1
        else:

            return ctrdms

    # chl_opt=1
    else:

        ctrdms = np.zeros([n_subs, n_chls, nts, nts, n_cons, n_cons], dtype=np.float)

        for sub in range(n_subs):
            for chl in range(n_chls):

                print("\nSubject "+str(sub+1)+" Channel "+str(chl+1)+":")

                ctrdms[sub, chl] = single_cal.ctrdm_cal(data[:, sub, chl], time_win=time_win, time_step=time_step)

        # chl_opt=1 & sub_opt=0
        if sub_opt == 0:

            return np.average(ctrdms, axis=0)

        # chl_opt=1 & sub_opt=1
        else:

            return ctrdms

# test
#data = np.random.rand(16, 40, 5, 50)
#from pyctrsa.plotting import ctrdm
#ctrdms = ctrdms_cal(data, sub_opt=1, chl_opt=0)
#ctrdms = np.average(ctrdms, axis=0)
#ctrdm.ctrdm_plot(ctrdms[1, 3])