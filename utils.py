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

    # 获取文件名中的某一项
    # checkpoints_path = os.path.join(basedir, 'Conv2')
    # filename_list = os.listdir(checkpoints_path)
    # filename_list.sort(key=lambda x:int(x.split('.')[0]))
    # f = open('acc.txt', 'w')
    # for filename in filename_list:
    #     # print(filename)
    #     acc = filename.split('_')[2][3::1]
    #     f.writelines(acc + '\n')
    #     # print(acc)

    # 绘图
    # f = open('acc.txt', 'r')
    # acc_curve = []
    # file = f.read().splitlines()
    # for filename in file:
    #     acc_curve.append(float(filename))
    # train_x = range(len(acc_curve))
    # train_y = acc_curve
    #
    # plt.xticks(fontproperties='Times New Roman', size=20)
    # plt.yticks(fontproperties='Times New Roman', size=20)
    # plt.plot(train_x, train_y, label='Transformer', color='r')
    #
    # plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 20})
    # plt.ylabel('识别率', fontsize=20)
    # plt.xlabel('循环次数', fontsize=20)
    # # plt.savefig("acc_curve.png", bbox_inches='tight')
    # plt.show()

    # 随机抽取100条记录
    # f = open('wrong.txt', 'r')
    # fw = open('100.txt', 'w')
    # raw_list = f.readlines()
    # random.shuffle(raw_list)
    # for i in range(100):
    #     fw.writelines(raw_list[i])
    # f.close()
    # fw.close()

    # 删除指定文件名的文件
    # fuhao_dir = os.path.join(basedir, 'fuhao')
    # src_dir = os.path.join(basedir, 'words')
    # f = open('words.txt', 'r')
    # fw = open('1.txt', 'w')
    # i = 0
    # filename_list = f.readlines()
    # for filename in filename_list:
    #     if filename.split(' ')[-1] != '-\n':
    #         fw.writelines(filename)
            # start = filename.split(' ')[0]
            # src = os.path.join(src_dir, f'{start}.png')
            # print(src)
            # shutil.move(src, fuhao_dir)
    # f.close()
    # fw.close()

    # 获取训练、验证、测试集文本
    # f_src = open('lexicon.txt', 'r')
    # f_dis = open('valid1.txt', 'w')
    # i = 0
    # with open('validationset1.txt', 'r') as f:
    #     f_list = f.read().splitlines()
    #     for file in f_src.readlines():
    #         if file.split(' ')[0][0:-3:] in f_list:
    #             # print(file)
    #             i = i + 1
    #             f_dis.writelines(file)
    # f_src.close()
    # print(i)

def CER(label, prediction):
    return nltk.edit_distance(label, prediction) / len(label)

def WER(label, prediction):
    if label == prediction:
        return 0.0
    else:
        return 1.0
    # return nltk.edit_distance(prediction.split(' '), label.split(' ')) / len(label.split(' '))

def read_from_dgrl(dgrl):
    if not os.path.exists(dgrl):
        print('DGRL not exis!')
        return

    dir_name, base_name = os.path.split(dgrl)
    label_dir = dir_name + '_label'
    image_dir = dir_name + '_images'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(dgrl, 'rb') as f:
        # 读取表头尺寸
        header_size = np.fromfile(f, dtype='uint8', count=4)
        header_size = sum([j << (i * 8) for i, j in enumerate(header_size)])
        # print(header_size)

        # 读取表头剩下内容，提取 code_length
        header = np.fromfile(f, dtype='uint8', count=header_size - 4)
        code_length = sum([j << (i * 8) for i, j in enumerate(header[-4:-2])])
        # print(code_length)

        # 读取图像尺寸信息，提取图像中行数量
        image_record = np.fromfile(f, dtype='uint8', count=12)
        height = sum([j << (i * 8) for i, j in enumerate(image_record[:4])])
        width = sum([j << (i * 8) for i, j in enumerate(image_record[4:8])])
        line_num = sum([j << (i * 8) for i, j in enumerate(image_record[8:])])
        print('图像尺寸:')
        print(height, width, line_num)

        # 读取每一行的信息
        for k in range(line_num):
            print(k + 1)

            # 读取该行的字符数量
            char_num = np.fromfile(f, dtype='uint8', count=4)
            char_num = sum([j << (i * 8) for i, j in enumerate(char_num)])
            print('字符数量:', char_num)

            # 读取该行的标注信息
            label = np.fromfile(f, dtype='uint8', count=code_length * char_num)
            label = [label[i] << (8 * (i % code_length)) for i in range(code_length * char_num)]
            label = [sum(label[i * code_length:(i + 1) * code_length]) for i in range(char_num)]
            label = [struct.pack('I', i).decode('gbk', 'ignore')[0] for i in label]
            print('合并前：', label)
            label = ''.join(label)
            label = ''.join(label.split(b'\x00'.decode()))  # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题
            print('合并后：', label)

            # 读取该行的位置和尺寸
            pos_size = np.fromfile(f, dtype='uint8', count=16)
            y = sum([j << (i * 8) for i, j in enumerate(pos_size[:4])])
            x = sum([j << (i * 8) for i, j in enumerate(pos_size[4:8])])
            h = sum([j << (i * 8) for i, j in enumerate(pos_size[8:12])])
            w = sum([j << (i * 8) for i, j in enumerate(pos_size[12:])])
            # print(x, y, w, h)

            # 读取该行的图片
            bitmap = np.fromfile(f, dtype='uint8', count=h * w)
            bitmap = np.array(bitmap).reshape(h, w)

            # 保存信息
            label_file = os.path.join(label_dir, base_name.replace('.dgrl', '_' + str(k) + '.txt'))
            with open(label_file, 'w') as f1:
                f1.write(label)
            bitmap_file = os.path.join(image_dir, base_name.replace('.dgrl', '_' + str(k) + '.jpg'))
            cv.imwrite(bitmap_file, bitmap)