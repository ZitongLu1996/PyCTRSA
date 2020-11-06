# -*- coding: utf-8

"""
@File       :   t_normalrdmbased.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.ctsimilarity.normalrdmbased import ctsimilarities_cal

class test_normalrdmbased(unittest.TestCase):

    def test_ctsimilarities_cal(self):

        RDMs = np.random.rand(20, 6, 6)
        CTSimilarities = ctsimilarities_cal(RDMs)
        self.assertEqual(CTSimilarities.shape[0], 20)
        self.assertEqual(len(CTSimilarities.shape), 3)

        RDMs = np.random.rand(5, 20, 6, 6)
        CTSimilarities = ctsimilarities_cal(RDMs)
        self.assertEqual(CTSimilarities.shape[0], 5)
        self.assertEqual(len(CTSimilarities.shape), 4)

        RDMs = np.random.rand(5, 4, 20, 6, 6)
        CTSimilarities = ctsimilarities_cal(RDMs)
        self.assertEqual(CTSimilarities.shape[0], 5)
        self.assertEqual(len(CTSimilarities.shape), 5)