# -*- coding: utf-8

"""
@File       :   ctrdm.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

' a module for plotting the Cross-Temporal Representational Dissimilarity Matrix (CTRDM) '

import numpy as np
import matplotlib.pyplot as plt


' a function for plotting the Cross-Temporal RDM '

def ctrdm_plot(CTRDM, percentile=False, lim=[0, 2], conditions=None, con_fontsize=12, cmap=None):

    """
    Plot the Cross-Temporal RDM

    Parameters
    ----------
    CTRDM : array or list [n_cons, n_cons]
        A Cross-Temporal RDM.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    percentile : bool True or False. Default is False.
        Rescale the values in RDM or not by displaying the percentile.
    conditions : string-array or string-list. Default is None.
        The labels of the conditions for plotting.
        conditions should contain n_cons strings, If conditions=None, the labels of conditions will be invisible.
    con_fontsize : int or float. Default is 12.
        The fontsize of the labels of the conditions for plotting.
    cmap : matplotlib colormap. Default is None.
        The colormap for RDM.
        If cmap=None, the ccolormap will be 'jet'.

    Notes
    -----
    Only when percentile=False, lim works.
    """

    # get the number of conditions
    cons = CTRDM.shape[0]

    # if cons=2, the RDM cannot be plotted.
    if cons == 2:
        print("The shape of CTRDM cannot be 2*2. Here PyCTRSA cannot plot this CTRDM.")

        return None

    # determine if it's a square
    a, b = np.shape(CTRDM)
    if a != b:
        return None

    if percentile == True:

        v = np.zeros([cons*cons, 2], dtype=np.float)
        for i in range(cons):
            for j in range(cons):
                v[i*cons+j, 0] = CTRDM[i, j]

        index = np.argsort(v[:, 0])
        m = 0
        for i in range(cons*cons):
            if i > 0:
                if v[index[i], 0] > v[index[i - 1], 0]:
                    m = m + 1
                v[index[i], 1] = m

        v[:, 0] = v[:, 1] * 100 / m

        for i in range(cons):
            for j in range(cons):
                CTRDM[i, j] = v[i * cons + j, 0]

        if cmap == None:
            plt.imshow(CTRDM, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(0, 100))
        else:
            plt.imshow(CTRDM, extent=(0, 1, 0, 1), cmap=cmap, clim=(0, 100))

        # plt.axis("off")
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        font = {'size': 18}
        cb.set_label("Dissimilarity (percentile)", fontdict=font)

    else:

        # plot the RDM
        min = lim[0]
        max = lim[1]
        if cmap == None:
            plt.imshow(CTRDM, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(min, max))
        else:
            plt.imshow(CTRDM, extent=(0, 1, 0, 1), cmap=cmap, clim=(min, max))

        # plt.axis("off")
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        font = {'size': 18}
        cb.set_label("Dissimilarity", fontdict=font)

    if conditions != None:
        step = float(1 / cons)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, conditions, fontsize=con_fontsize, rotation=30, ha="right")
        plt.yticks(y, conditions, fontsize=con_fontsize)
    else:
        plt.axis("off")

    plt.xlabel("Time 1", fontsize=18)
    plt.ylabel("Time 2", fontsize=18)

    plt.show()