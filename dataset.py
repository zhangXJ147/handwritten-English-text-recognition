import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.config import train_config as config
from src.augment import distort, stretch, perspective

class IAMDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-\',.*"+?/'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, transform=None, img_height=32, img_width=96):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for line in fr.readlines():
                mapping[line.split(' ')[0]] = line.strip().split(' ')[-1]  # strip()只能删除开头或是结尾的字符，不能删除中间部分的字符。

        paths_file = None
        if mode == 'train':
            paths_file = 'train.txt'
        elif mode == 'dev':
            paths_file = 'valid.txt'
        elif mode == 'test':
            paths_file = 'test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                # path = os.path.join(root_dir, path)
                text = mapping[index_str]
                paths.append(path)
                texts.append(text)
        return paths, texts


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        if self.transform is not None:
            image = self.transform(image)
            image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
            image = np.array(image)

            num = random.random()
            if num < 0.3:
                image = distort(image, 4)
            elif num > 0.7:
                image = perspective(image)
            else:
                image = stretch(image, 4)
        else:
            image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
            image = np.array(image)

        image = image.reshape((self.img_height, self.img_width, 1))
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(0.485, 0.229)(image)

        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            # 如果DataLoader不设置collate_fn,则此处返回值为迭代DataLoader时取到的值
            return image, target, target_length
        else:
            return image


def IAM_collate_fn(batch):
    # zip(*batch)拆包
    images, targets, target_lengths = zip(*batch)
    # stack就是向量堆叠的意思。一定是扩张一个维度，然后在扩张的维度上，把多个张量纳入进仅一个张量。想象向上摞面包片，摞的操作即是stack，0轴即按块stack
    images = torch.stack(images, 0)
    # cat是指向量拼接的意思。一定不扩张维度，想象把两个长条向量cat成一个更长的向量。
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    # 此处返回的数据即是train_loader每次取到的数据，迭代train_loader，每次都会取到三个值，即此处返回值。
    return images, targets, target_lengths

if __name__ == '__main__':
    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']
    train_batch_size = config['train_batch_size']
    cpu_workers = config['cpu_workers']

    train_transform = transforms.RandomChoice([
        transforms.RandomAffine(degrees=(-2, 2), translate=(0, 0), scale=(0.9, 1),
                                shear=5, resample=False, fillcolor=255),
        transforms.RandomAffine(degrees=(-3, 3), translate=(0, 0.2), scale=(0.9, 1),
                                shear=5, resample=False, fillcolor=255),
    ])

    train_dataset = IAMDataset(root_dir=data_dir, mode='train', transform=train_transform,
                                    img_height=img_height, img_width=img_width)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=IAM_collate_fn)

    train_data = train_dataset.__getitem__(136)
    print(f'train_data的类型是：{type(train_data)}')
    print(f'train_data的长度是：{len(train_data)}')