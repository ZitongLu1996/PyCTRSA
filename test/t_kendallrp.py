# -*- coding: utf-8

"""
@File       :   t_kendallrp.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.similarity.kendallrp import kendallrp_cal

class test_kendallrp(unittest.TestCase):

    def test_kendallrp_cal(self):

        CTRDMs1 = np.random.rand(16, 16)
        CTRDMs2 = np.random.rand(16, 16)
        CTSimilarities = kendallrp_cal(CTRDMs1, CTRDMs2)
        self.assertEqual(len(CTSimilarities), 2)