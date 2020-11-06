# -*- coding: utf-8

"""
@File       :   progressbar.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import sys

def show_progressbar(str, cur, total=100):

    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write(str + ": [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()