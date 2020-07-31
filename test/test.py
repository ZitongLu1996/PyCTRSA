# -*- coding: utf-8

"""
@File       :   test.py
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

url = 'https://attachment.zhaokuangshi.cn/BaeLuck_2018jn_data.zip'
filename = 'BaeLuck_2018jn_data.zip'
data_dir = '/Users/zitonglu/Downloads/PyCTRSA-master-2/data/'
filepath = data_dir + filename

"""  Section 1: Download Data  """

"""exist = os.path.exists(filepath)
if exist == False:
    os.makedirs(data_dir)
    urllib.request.urlretrieve(url, filepath, schedule)
    print('Download completes!')
elif exist == True:
    print('Data already exists!')

unzipfile(filepath, data_dir)"""


"""  Section 2: Data preprocessing for further calculation  """

subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]

pre_data(subs, data_dir)


"""  Section 3: Calculating Cross-Temporal RDMs  """

# calculate the CTRDMs for a single channel & a single subject

# here we choose the ERP position data of 1st channel of sub201, firstly
f = h5py.File(data_dir+'data_for_CTRSA/ERP/201.h5', 'r')
data_pos_ERP_sub201chl1 = np.array(f['pos'])[:, :, 0]
f.close()

# avgerage the trials
data_pos_ERP_sub201chl1 = np.average(data_pos_ERP_sub201chl1, axis=1)
# calculate the CTRDMs
# here, time_win=5 and time_step=5 (5 time-points corresponding to 10ms in this Experiment)
CTRDM_pos_ERP_sub201chl1 = single_cal.ctrdm_cal(data_pos_ERP_sub201chl1, time_win=5, time_step=5)

# save the CTRDMs
f = h5py.File('test_resutls/CTRDM_pos_ERP_sub201chl1.h5', 'w')
f.create_dataset('CTRDMs', data=CTRDM_pos_ERP_sub201chl1)
f.close()

# plot one of the CTRDMs (time 1: 10 ms, time 2: 30 ms, the time of stimulus onset is 0 ms)
conditions = ["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°", "180°",
              "202.5°", "225°", "247.5°", "270°", "292.5°", "315°", "337.5°"]
ctrdm.ctrdm_plot(CTRDM_pos_ERP_sub201chl1[30, 40])


# calculate the CTRDMs for all channels & subjects

# here we use the ERP position data of all channels & subjects
nsubs = len(subs)
data_pos_ERP = np.zeros([16, nsubs, 40, 27, 500], dtype=np.float)
subindex = 0
for sub in subs:
    print('Loading data of sub'+sub)
    f = h5py.File(data_dir+'data_for_CTRSA/ERP/'+sub+'.h5', 'r')
    ori_subdata = np.array(f['pos'])
    f.close()
    data_pos_ERP[:, subindex] = ori_subdata

# calculate the CTRDMs
CTRDM_pos_ERP = multi_cal.ctrdms_cal(data_pos_ERP, sub_opt=1, chl_opt=0, time_win=5, time_step=5)

# save the CTRDMs
f = h5py.File('test_resutls/CTRDM_pos_ERP.h5', 'w')
f.create_dataset('CTRDMs', data=CTRDM_pos_ERP)
f.close()

# plot one of the CTRDMs (time 1: 10 ms, time 2: 20 ms, the time of stimulus onset is 0 ms) averaging the subjects
ctrdm.ctrdm_plot(np.average(CTRDM_pos_ERP, axis=0)[30, 35])


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
ctrdm.ctrdm_plot(pos_model_RDM, percentile=True, conditions=conditions)

# calculate the CTSimilarities between CTRDMs and Position-Coding RDM
CTSim_pos_ERP = fitctrdm.ctsimilarities_cal(CTRDM_pos_ERP, pos_model_RDM)

# save the CTSimilarities
f = h5py.File('test_resutls/CTSimilarities_pos_ERP.h5', 'w')
f.create_dataset('CTSimilarities', data=CTSim_pos_ERP)
f.close()

# plot the time-by-time decoding results
tbytsimilarities.tbytsimilarities_plot(CTSim_pos_ERP, start_time=-0.5, end_time=1.5, color='orange', lim=[-0.1, 0.5])

# plot the Cross-Temporal decoding results
ctsimilarities.ctsimilarities_plot(CTSim_pos_ERP, start_time=-0.5, end_time=1.5, cmap='bwr', lim=[-0.08, 0.08])