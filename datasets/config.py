'''
@Author: Jeffery Sheng (Zhenfei Sheng)
@Time:   2020/5/21 12:04
@File:   config.py
'''


textline = '/data/disk7/private/szf/Datasets/ICDAR2015/train.txt'
alphabet = 'datasets/alphabets/enAlphaNumPunc90.txt'
# mode = 'train'
augmentation = True
input_h = 32
input_w = 160 # base input width of img, it will determine the cache group
mean = 0.588
std = 0.193
batch_size = 4
shuffle = True
num_workers = 4
num_cache = 10 # num of data caches
