# -*- coding: utf-8

"""
@File       :   preprocess_data.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import os
import scipy.io as sio
import h5py
from pyctrsa.util.progressbar import show_progressbar

def pre_data(subs, data_dir):

    newdata_dir = data_dir + 'data_for_CTRSA/'

    if os.path.exists(newdata_dir) == False:
        os.makedirs(newdata_dir)
    if os.path.exists(newdata_dir + 'ERP/') == False:
        os.makedirs(newdata_dir + 'ERP/')
    if os.path.exists(newdata_dir + 'Alpha/') == False:
        os.makedirs(newdata_dir + 'Alpha/')

    n = len(subs)
    subindex = 0

    for sub in subs:
        data = sio.loadmat(data_dir + "data/ERP" + sub + ".mat")["filtData"][:, :, 250:]
        # data.shape: n_trials, n_channels, n_times

        ori_label = np.loadtxt(data_dir + "labels/ori_" + sub + ".txt")[:, 1]
        pos_label = np.loadtxt(data_dir + "labels/pos_" + sub + ".txt")[:, 1]

        ori_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)
        pos_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)

        ori_labelindex = np.zeros([16], dtype=np.int)
        pos_labelindex = np.zeros([16], dtype=np.int)

        for i in range(640):
            label = int(ori_label[i])
            ori_subdata[label, ori_labelindex[label]] = data[i]
            ori_labelindex[label] = ori_labelindex[label] + 1
            label = int(pos_label[i])
            pos_subdata[label, pos_labelindex[label]] = data[i]
            pos_labelindex[label] = pos_labelindex[label] + 1

        f = h5py.File(newdata_dir + "ERP/" + sub + ".h5", "w")
        f.create_dataset("ori", data=ori_subdata)
        f.create_dataset("pos", data=pos_subdata)
        f.close()

        data = sio.loadmat(data_dir + "data/Alpha" + sub + ".mat")["filtData"][:, :, 250:]
        # data.shape: n_trials, n_channels, n_times

        ori_label = np.loadtxt(data_dir + "labels/ori_" + sub + ".txt")[:, 1]
        pos_label = np.loadtxt(data_dir + "labels/pos_" + sub + ".txt")[:, 1]

        ori_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)
        pos_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)

        ori_labelindex = np.zeros([16], dtype=np.int)
        pos_labelindex = np.zeros([16], dtype=np.int)

        for i in range(640):
            label = int(ori_label[i])
            ori_subdata[label, ori_labelindex[label]] = data[i]
            ori_labelindex[label] = ori_labelindex[label] + 1
            label = int(pos_label[i])
            pos_subdata[label, pos_labelindex[label]] = data[i]
            pos_labelindex[label] = pos_labelindex[label] + 1

        subindex = subindex + 1
        percent = subindex/n*100
        show_progressbar("Preprocessing", percent)

        f = h5py.File(newdata_dir + "Alpha/" + sub + ".h5", "w")
        f.create_dataset("ori", data=ori_subdata)
        f.create_dataset("pos", data=pos_subdata)
        f.close()
