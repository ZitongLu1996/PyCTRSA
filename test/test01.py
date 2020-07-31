# -*- coding: utf-8

"""
@File       :   test01.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""
import os
import numpy as np
from six.moves import urllib
from pyctrsa.util.progressbar import show_progressbar
from pyctrsa.util.download_data import schedule
from pyctrsa.util.unzip_data import unzipfile
from pyctrsa.util.preprocess_data import pre_data
from pyctrsa.ctrdm import single_cal, multi_cal
from pyctrsa.plotting import ctrdm, tbytsimilarities, ctsimilarities
from pyctrsa.ctsimilarity import fitctrdm
import h5py

"""  Section 4: Calculating and Plotting Cross-Temporal Similarities  """

# establish a Position-Coding RDM
# In this model RDM, the representational dissimilarities between two positions are larger
# when the difference between these two positions are large and the dissimilarities between
# two positions are smaller when the difference between these two positions are small.
pos_model_RDM = np.zeros([16, 16], dtype=np.float)
for i in range(16):
    for j in range(16):
        diff = np.abs(i-j)
        if diff <= 8:
            pos_model_RDM[i, j] = diff/8
        else:
            pos_model_RDM[i, j] = (16-diff)/8

# plot the Position-Coding RDM
conditions = ["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°", "180°",
              "202.5°", "225°", "247.5°", "270°", "292.5°", "315°", "337.5°"]
ctrdm.ctrdm_plot(pos_model_RDM, percentile=True, conditions=conditions)

# calculate the CTSimilarities between CTRDMs and Position-Coding RDM
f = h5py.File('test_resutls/CTRDM_pos_ERP.h5', 'r')
CTRDM_pos_ERP = np.array(f['CTRDMs'])
f.close()
CTSim_pos_ERP = fitctrdm.ctsimilarities_cal(CTRDM_pos_ERP, pos_model_RDM)

# save the CTSimilarities
f = h5py.File('test_resutls/CTSimilarities_pos_ERP.h5', 'w')
f.create_dataset('CTSimilarities', data=CTSim_pos_ERP)
f.close()

# plot the time-by-time decoding results
tbytsimilarities.tbytsimilarities_plot(CTSim_pos_ERP, start_time=-0.5, end_time=1.5, color='orange', lim=[-0.1, 0.5])

# plot the Cross-Temporal decoding results
ctsimilarities.ctsimilarities_plot(CTSim_pos_ERP, start_time=-0.5, end_time=1.5, cmap='bwr', lim=[-0.08, 0.08])