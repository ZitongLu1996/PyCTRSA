# -*- coding: utf-8

"""
@File       :   t_multi_cal.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.ctrdm.multi_cal import ctrdms_cal

class test_multi_cal(unittest.TestCase):

    def test_ctrdms_cal(self):

        data = np.random.rand(10, 8, 16, 20)
        CTRDMs = ctrdms_cal(data, chl_opt=1, time_win=10, time_step=5)
        self.assertEqual(CTRDMs.shape[0], 8)
        self.assertEqual(len(CTRDMs.shape), 6)

        CTRDMs = ctrdms_cal(data, chl_opt=0, time_win=10, time_step=5)
        self.assertEqual(CTRDMs.shape[0], 8)
        self.assertEqual(len(CTRDMs.shape), 5)
