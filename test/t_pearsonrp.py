# -*- coding: utf-8

"""
@File       :   t_pearsonrp.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.similarity.pearsonrp import pearsonrp_cal

class test_pearsonrp(unittest.TestCase):

    def test_pearsonrp_cal(self):

        CTRDMs1 = np.random.rand(16, 16)
        CTRDMs2 = np.random.rand(16, 16)
        CTSimilarities = pearsonrp_cal(CTRDMs1, CTRDMs2)
        self.assertEqual(len(CTSimilarities), 2)