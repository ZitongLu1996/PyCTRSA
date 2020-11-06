# -*- coding: utf-8

"""
@File       :   download_data.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

from six.moves import urllib
import sys
from pyctrsa.util.progressbar import show_progressbar

def schedule(blocknum,blocksize,totalsize):

    if totalsize == 0:
        percent = 0
    else:
        percent = blocknum * blocksize / totalsize
    if percent > 1.0:
        percent = 1.0
    percent = percent * 100
    show_progressbar("Downloading", percent)

# test
#url = 'https://attachment.zhaokuangshi.cn/BaeLuck_2018jn_data.zip'
#filename = 'BaeLuck_2018jn_data.zip'
#data_dir = '/Users/zitonglu/Downloads/'
#filepath = data_dir + filename

#urllib.request.urlretrieve(url, filepath, schedule)
#print('Download completes!')

