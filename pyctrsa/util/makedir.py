# -*- coding: utf-8

"""
@File       :   makedir.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import os

def mkdir(path):

    path = path.strip()
    path = path.rstrip("/")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

        print(path + ' is created successfully!')
        return True
    else:
        print(path + ' already exists')
        return False

# test codes
#path = '../data/'
#mkdir(path)