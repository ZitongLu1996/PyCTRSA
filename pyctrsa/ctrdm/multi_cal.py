# -*- coding: utf-8

"""
@File       :   multi_cal.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for calculating Cross-Temporal RDMs for multi-channels data '

import numpy as np
from pyctrsa.ctrdm import single_cal

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
    CTRDMs : array [int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1, n_conditions, n_conditions]
        Cross-Temporal RDMs.
    """

    n_cons, n_subs, n_chls, n_ts = np.shape(data)

    nts = int((n_ts - time_win) / time_step) + 1

    # chl_opt=0
    if chl_opt == 0:

        data = np.average(data, axis=2)

        ctrdms = np.zeros([n_subs, nts, nts, n_cons, n_cons], dtype=np.float)

        for sub in range(n_subs):

            print("\nSubject "+str(sub+1)+":")

            ctrdms[sub] = single_cal.ctrdm_cal(data[:, sub], time_win=time_win, time_step=time_step)

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
data = np.random.rand(10, 10, 20, 50)
from pyctrsa.plotting import ctrdm
ctrdms = ctrdms_cal(data, sub_opt=1, chl_opt=0)
ctrdms = np.average(ctrdms, axis=0)
ctrdm.ctrdm_plot(ctrdms[1, 3])