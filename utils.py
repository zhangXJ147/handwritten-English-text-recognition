import glob, os
import random
import nltk
from tqdm import tqdm
import struct
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import shutil

# from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']

if __name__ == "__main__":

    basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = f'{basedir}/data/words'

    total_images_path_list = [image for image in glob.iglob(os.path.join(raw_dir, '*.png'))]

    for image in tqdm(total_images_path_list):
        # tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        title, ext = os.path.basename(image).split('.')
        label = title.split('_')[-1]
        num = random.random()
        if num <= 0.1:
            with open('../data/labels/test.txt', 'a') as f:
                f.write(f'{image} {label}\n')
        elif 0.1 < num <= 0.2:
            with open('../data/labels/valid.txt', 'a') as f:
                f.write(f'{image} {label}\n')
        else:
            with open('../data/labels/train.txt', 'a') as f:
                f.write(f'{image} {label}\n')

def CER(label, prediction):
    return nltk.edit_distance(label, prediction) / len(label)

def WER(label, prediction):
    if label == prediction:
        return 0.0
    else:
        return 1.0
    # return nltk.edit_distance(prediction.split(' '), label.split(' ')) / len(label.split(' '))
