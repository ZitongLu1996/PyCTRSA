# -*- coding: utf-8

"""
@File       :   t_normaldatabased.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.ctsimilarity.normaldatabased import ctsimilarities_cal

class test_normaldatabased(unittest.TestCase):

    def test_ctsimilarities_cal(self):

        data1 = np.random.rand(8, 16, 20)
        data2 = np.random.rand(8, 16, 20)
        CTSimilarities = ctsimilarities_cal(data1=data1, data2=data2, sub_opt=1, chl_opt=1)
        self.assertEqual(CTSimilarities.shape[0], 8)
        self.assertEqual(len(CTSimilarities.shape), 5)


        CTSimilarities = ctsimilarities_cal(data1=data1, data2=data2, sub_opt=1, chl_opt=0)
        self.assertEqual(CTSimilarities.shape[0], 8)
        self.assertEqual(len(CTSimilarities.shape), 4)


        CTSimilarities = ctsimilarities_cal(data1=data1, data2=data2, sub_opt=0, chl_opt=1)
        self.assertEqual(CTSimilarities.shape[0], 16)
        self.assertEqual(len(CTSimilarities.shape), 4)


        CTSimilarities = ctsimilarities_cal(data1=data1, data2=data2, sub_opt=0, chl_opt=0)
        self.assertEqual(CTSimilarities.shape[0], 3)
        self.assertEqual(len(CTSimilarities.shape), 3)