# -*- coding: utf-8

"""
@File       :   t_fitctrdm.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.ctsimilarity.fitctrdm import ctsimilarities_cal

class test_fitctrdm_cal(unittest.TestCase):

    def test_ctsimilarities_cal(self):

        CTRDMs = np.random.rand(10, 10, 16, 16)
        Model_RDM = np.random.rand(16, 16)
        CTSimilarities = ctsimilarities_cal(CTRDMs, Model_RDM)
        self.assertEqual(CTSimilarities.shape[0], 10)
        self.assertEqual(len(CTSimilarities.shape), 3)

        CTRDMs = np.random.rand(8, 10, 10, 16, 16)
        CTSimilarities = ctsimilarities_cal(CTRDMs, Model_RDM)
        self.assertEqual(CTSimilarities.shape[0], 8)
        self.assertEqual(len(CTSimilarities.shape), 4)