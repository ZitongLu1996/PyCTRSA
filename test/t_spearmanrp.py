# -*- coding: utf-8

"""
@File       :   t_spearmanrp.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.similarity.spearmanrp import spearmanrp_cal

class test_spearmanrp(unittest.TestCase):

    def test_spearmanrp_cal(self):

        CTRDMs1 = np.random.rand(16, 16)
        CTRDMs2 = np.random.rand(16, 16)
        CTSimilarities = spearmanrp_cal(CTRDMs1, CTRDMs2)
        self.assertEqual(len(CTSimilarities), 2)