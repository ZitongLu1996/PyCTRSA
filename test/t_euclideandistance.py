# -*- coding: utf-8

"""
@File       :   t_euclideandistance.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.similarity.euclideandistance import euclideandistance_cal

class test_euclideandistance(unittest.TestCase):

    def test_euclideandistance_cal(self):

        CTRDMs1 = np.random.rand(16, 16)
        CTRDMs2 = np.random.rand(16, 16)
        CTSimilarities = euclideandistance_cal(CTRDMs1, CTRDMs2)
        self.assertNotEqual(CTSimilarities, None)