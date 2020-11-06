# -*- coding: utf-8

"""
@File       :   t_single_cal.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.ctrdm.single_cal import ctrdm_cal

class test_single_cal(unittest.TestCase):

    def test_ctrdms_cal(self):

        data = np.random.rand(10, 20)
        CTRDMs = ctrdm_cal(data, time_win=10, time_step=5)
        self.assertEqual(CTRDMs.shape[0], 3)
        self.assertEqual(len(CTRDMs.shape), 4)