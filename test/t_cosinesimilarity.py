# -*- coding: utf-8

"""
@File       :   t_cosinesimilarity.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import unittest
from pyctrsa.similarity.cosinesimilarity import cosinesimilarity_cal

class test_cosinesimilarity(unittest.TestCase):

    def test_cosinesimilarity_cal(self):

        CTRDMs1 = np.random.rand(16, 16)
        CTRDMs2 = np.random.rand(16, 16)
        CTSimilarities = cosinesimilarity_cal(CTRDMs1, CTRDMs2)
        self.assertNotEqual(CTSimilarities, None)