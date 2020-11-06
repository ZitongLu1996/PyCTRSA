# -*- coding: utf-8

"""
@File       :   tbytsimilarities.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for plotting the time-by-time similarities '

import numpy as np
from neurora.stuff import permutation_test
import matplotlib.pyplot as plt
from neurora.rsa_plot import plot_corrs_hotmap


' a function for plotting the time-by-time similarities averaging all subjects '

def tbytsimilarities_plot(CTSimilarities, start_time=0, end_time=1, color='r', lim=[-0.1, 0.8]):

    """
    Plot the time-by-time Similarities averaging all subjects

    Parameters
    ----------
    CTSimilarities : array
        The Cross-Temporal Similarities.
        The size of CTSimilarities should be [n_subs, n_ts, n_ts] or [n_subs, n_ts, n_ts, 2]. n_subs, n_ts represent the
        number of subjects and number of time-points. 2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    color : matplotlib color or None. Default is 'r'.
        The color for the curve.
    lim : array or list [min, max]. Default is [-0.1, 0.8].
        The corrs view lims.
    """

    n = len(np.shape(CTSimilarities))

    minlim = lim[0]
    maxlim = lim[1]

    if n == 4:
        CTSimilarities = CTSimilarities[:, :, :, 0]

    nsubs = np.shape(CTSimilarities)[0]
    nts = np.shape(CTSimilarities)[1]

    tstep = float((end_time-start_time)/nts)

    sim = np.zeros([nsubs, nts], dtype=np.float)

    for sub in range(nsubs):
        for t in range(nts):

            sim[sub, t] = CTSimilarities[sub, t, t]

    for sub in range(nsubs):
        for t in range(nts):

            if t<=1:
                sim[sub, t] = np.average(sim[sub, :t+3])
            if t>1 and t<(nts-2):
                sim[sub, t] = np.average(sim[sub, t-2:t+3])
            if t>=(nts-2):
                sim[sub, t] = np.average(sim[sub, t-2:])

    avg = np.average(sim, axis=0)
    err = np.zeros([nts], dtype=np.float)

    for t in range(nts):
        err[t] = np.std(sim[:, t], ddof=1)/np.sqrt(nsubs)

    ps = np.zeros([nts], dtype=np.float)
    chance = np.full([nsubs], 0)

    for t in range(nts):
        ps[t] = permutation_test(sim[:, t], chance)
        if ps[t] < 0.05 and avg[t] > 0:
            plt.plot(t*tstep+start_time, maxlim*0.9, 's', color=color, alpha=1)
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [0]
            ymax = [avg[t]-err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor=color, alpha=0.1)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines['bottom'].set_position(('data', 0))

    x = np.arange(start_time+0.5*tstep, end_time+0.5*tstep, tstep)
    plt.fill_between(x, avg + err, avg - err, facecolor=color, alpha=0.8)
    plt.ylim(minlim, maxlim)
    plt.xlim(start_time, end_time)
    plt.tick_params(labelsize=12)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Representational Similarity", fontsize=16)
    plt.show()


' a function for plotting the time-by-time similarities for all subjects '

def tbytsimilarities_plot_bysubject(CTSimilarities, start_time=0, end_time=1, figsize=None, cmap='bwr', lim=[-0.1, 0.8], sub_ids=None):
    """
    Plot the time-by-time Similarities for all subjects

    Parameters
    ----------
    CTSimilarities : array
        The Cross-Temporal Similarities.
        The size of CTSimilarities should be [n_subs, n_ts, n_ts] or [n_subs, n_ts, n_ts, 2]. n_subs, n_ts represent the
        number of subjects and number of time-points. 2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is 'bwr'.
        The colormap for the figure.
    sub_ids : string-array or string-list. Default is None.
        The subject IDs.
    """

    n = len(np.shape(CTSimilarities))

    if n == 4:
        CTSimilarities = CTSimilarities[:, :, :, 0]

    nsubs = np.shape(CTSimilarities)[0]
    nts = np.shape(CTSimilarities)[1]

    tstep = float((end_time - start_time) / nts)

    sim = np.zeros([nsubs, nts], dtype=np.float)

    for sub in range(nsubs):
        for t in range(nts):
            sim[sub, t] = CTSimilarities[sub, t, t]

    for sub in range(nsubs):
        for t in range(nts):

            if t <= 1:
                sim[sub, t] = np.average(sim[sub, :t + 3])
            if t > 1 and t < (nts - 2):
                sim[sub, t] = np.average(sim[sub, t - 2:t + 3])
            if t >= (nts - 2):
                sim[sub, t] = np.average(sim[sub, t - 2:])

    plot_corrs_hotmap(sim, chllabels=sub_ids, time_unit=[start_time, tstep], lim=lim, figsize=figsize, cmap=cmap)


' a function for plotting the time-by-time similarities for all channels '

def tbytsimilarities_plot_bychannel(CTSimilarities, start_time=0, end_time=1, figsize=None, cmap='bwr', lim=[-0.1, 0.8], chl_ids=None):
    """
    Plot the time-by-time Similarities for all channels

    Parameters
    ----------
    CTSimilarities : array
        The Cross-Temporal Similarities.
        The size of CTSimilarities should be [n_subs, n_ts, n_ts] or [n_subs, n_ts, n_ts, 2]. n_subs, n_ts represent the
        number of subjects and number of time-points. 2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is 'bwr'.
        The colormap for the figure.
    chl_ids : string-array or string-list. Default is None.
        The channel IDs.
    """

    n = len(np.shape(CTSimilarities))

    if n == 4:
        CTSimilarities = CTSimilarities[:, :, :, 0]

    nchls = np.shape(CTSimilarities)[0]
    nts = np.shape(CTSimilarities)[1]

    tstep = float((end_time - start_time) / nts)

    sim = np.zeros([nchls, nts], dtype=np.float)

    for chl in range(nchls):
        for t in range(nts):
            sim[chl, t] = CTSimilarities[chl, t, t]

    for chl in range(nchls):
        for t in range(nts):

            if t <= 1:
                sim[chl, t] = np.average(sim[chl, :t + 3])
            if t > 1 and t < (nts - 2):
                sim[chl, t] = np.average(sim[chl, t - 2:t + 3])
            if t >= (nts - 2):
                sim[chl, t] = np.average(sim[chl, t - 2:])

    plot_corrs_hotmap(sim, chllabels=chl_ids, time_unit=[start_time, tstep], lim=lim, figsize=figsize, cmap=cmap)

