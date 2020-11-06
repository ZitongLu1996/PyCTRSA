# -*- coding: utf-8

"""
@File       :   ctsimilarities_plot.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a modulate of plotting the Cross-Temporal Similarities '

import numpy as np
import matplotlib.pyplot as plt


' a function for plotting the Cross-Temporal Similarities averaging all subjects'

def ctsimilarities_plot(CTSimilarities, start_time=0, end_time=1, cmap='bwr', lim=[-0.1, 0.1]):

    """
    Plot the Cross-Temporal Similarities averaging all subjects

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
    cmap : matplotlib colormap or None. Default is 'bwr'.
        The colormap for the figure.
    lim : array or list [min, max]. Default is [-0.1, 0.1].
        The similarity view lims.
    """

    n = len(np.shape(CTSimilarities))

    if n == 4:
        CTSimilarities = CTSimilarities[:, :, :, 0]

    avg_sim = np.average(CTSimilarities, axis=0)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    minlim = lim[0]
    maxlim = lim[1]

    plt.imshow(avg_sim, extent=(start_time, end_time, start_time, end_time), origin='low', cmap=cmap,
               clim=(minlim, maxlim))

    cb = plt.colorbar(ticks=[minlim, maxlim])
    cb.ax.tick_params(labelsize=12)
    font = {'size': 15,}
    cb.set_label('Representational Similarity', fontdict=font)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Training Time-point (s)", fontsize=16)
    plt.ylabel("Test Time-point (s)", fontsize=16)
    plt.show()


' a function for plotting the Cross-Temporal Similarities for all subjects'

def ctsimilarities_plot_bysub(CTSimilarities, start_time=0, end_time=1, cmap='bwr', lim=[-0.1, 0.1], sub_ids=None):

    """
    Plot the Cross-Temporal Similarities for all subjects

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
    cmap : matplotlib colormap or None. Default is 'bwr'.
        The colormap for the figure.
    lim : array or list [min, max]. Default is [-0.1, 0.1].
        The similarity view lims.
    sub_ids : string-array or string-list. Default is None.
        The subject IDs.
    """

    n = len(np.shape(CTSimilarities))

    if n == 4:
        CTSimilarities = CTSimilarities[:, :, :, 0]

    nsubs = np.shape(CTSimilarities)[0]

    for sub in range(nsubs):

        subsim = CTSimilarities[sub]

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)

        minlim = lim[0]
        maxlim = lim[1]

        plt.imshow(subsim, extent=(start_time, end_time, start_time, end_time), origin='low', cmap=cmap,
                    clim=(minlim, maxlim))

        cb = plt.colorbar(ticks=[minlim, maxlim])
        cb.ax.tick_params(labelsize=12)
        font = {'size': 15,}
        cb.set_label('Representational Similarity', fontdict=font)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Training Time-point (s)", fontsize=16)
        plt.ylabel("Test Time-point (s)", fontsize=16)
        if sub_ids != None:
            plt.title("Subject "+sub_ids[sub], fontsize=18)
        else:
            plt.title("Subject "+str(sub+1), fontsize=18)
        plt.show()


' a function for plotting the Cross-Temporal Similarities for all channels'

def ctsimilarities_plot_bysub(CTSimilarities, start_time=0, end_time=1, cmap='bwr', lim=[-0.1, 0.1], chl_ids=None):

    """
    Plot the Cross-Temporal Similarities for all channels

    Parameters
    ----------
    CTSimilarities : array
        The Cross-Temporal Similarities.
        The size of CTSimilarities should be [n_channels, n_ts, n_ts] or [n_channels, n_ts, n_ts, 2]. n_channels, n_ts
        represent the number of channels and number of time-points. 2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    cmap : matplotlib colormap or None. Default is 'bwr'.
        The colormap for the figure.
    lim : array or list [min, max]. Default is [-0.1, 0.1].
        The similarity view lims.
    chl_ids : string-array or string-list. Default is None.
        The channel IDs.
    """

    n = len(np.shape(CTSimilarities))

    if n == 4:
        CTSimilarities = CTSimilarities[:, :, :, 0]

    nchls = np.shape(CTSimilarities)[0]

    for chl in range(nchls):

        chlsim = CTSimilarities[chl]

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)

        minlim = lim[0]
        maxlim = lim[1]

        plt.imshow(chlsim, extent=(start_time, end_time, start_time, end_time), origin='low', cmap=cmap,
                    clim=(minlim, maxlim))

        cb = plt.colorbar(ticks=[minlim, maxlim])
        cb.ax.tick_params(labelsize=12)
        font = {'size': 15,}
        cb.set_label('Representational Similarity', fontdict=font)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Training Time-point (s)", fontsize=16)
        plt.ylabel("Test Time-point (s)", fontsize=16)
        if chl_ids != None:
            plt.title("Channel " + chl_ids[chl], fontsize=18)
        else:
            plt.title("Channel " + str(chl + 1), fontsize=18)
        plt.show()


' a function for plotting the Cross-Temporal Similarities for all subjects and channels'

def ctsimilarities_plot_bysub(CTSimilarities, start_time=0, end_time=1, cmap='bwr', lim=[-0.1, 0.1], sub_ids=None, chl_ids=None):

    """
    Plot the Cross-Temporal Similarities for all subjects and channels

    Parameters
    ----------
    CTSimilarities : array
        The Cross-Temporal Similarities.
        The size of CTSimilarities should be [n_subs, n_channels, n_ts, n_ts] or [n_subs, n_channels, n_ts, n_ts, 2].
        n_subs, n_channels, n_ts represent the number of subjects, the number of channels and number of time-points.
        2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    cmap : matplotlib colormap or None. Default is 'bwr'.
        The colormap for the figure.
    lim : array or list [min, max]. Default is [-0.1, 0.1].
        The similarity view lims.
    sub_ids : string-array or string-list. Default is None.
        The subject IDs.
    chl_ids : string-array or string-list. Default is None.
        The channel IDs.
    """

    n = len(np.shape(CTSimilarities))

    if n == 5:
        CTSimilarities = CTSimilarities[:, :, :, :, 0]

    nsubs = np.shape(CTSimilarities)[0]
    nchls = np.shape(CTSimilarities)[1]

    for sub in range(nsubs):
        for chl in range(nchls):

            subchlsim = CTSimilarities[sub, chl]

            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_linewidth(2)

            minlim = lim[0]
            maxlim = lim[1]

            plt.imshow(subchlsim, extent=(start_time, end_time, start_time, end_time), origin='low', cmap=cmap,
                        clim=(minlim, maxlim))

            cb = plt.colorbar(ticks=[minlim, maxlim])
            cb.ax.tick_params(labelsize=12)
            font = {'size': 15,}
            cb.set_label('Representational Similarity', fontdict=font)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel("Training Time-point (s)", fontsize=16)
            plt.ylabel("Test Time-point (s)", fontsize=16)
            if sub_ids != None and chl_ids != None:
                plt.title("Subject" + sub_ids[sub] + "Channel " + chl_ids[chl], fontsize=18)
            elif sub_ids != None and chl_ids == None:
                plt.title("Subject" + sub_ids[sub] + "Channel " + str(chl+1), fontsize=18)
            elif sub_ids == None and chl_ids != None:
                plt.title("Subject" + str(sub+1) + "Channel " + chl_ids[chl], fontsize=18)
            else:
                plt.title("Subject" + str(sub+1) + "Channel " + str(chl+1), fontsize=18)
            plt.show()


