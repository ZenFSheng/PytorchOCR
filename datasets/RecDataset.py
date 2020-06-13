'''
@Author: Jeffery Sheng (Zhenfei Sheng)
@Time:   2020/6/8 19:44
@File:   RecDataset.py
'''

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import StrLabelConverter
from util_scripts.CreateRecAug import cv2pil, pil2cv, RandomBrightness, RandomContrast, \
                                      RandomLine, RandomSharpness, Compress, Rotate, \
                                      Blur, MotionBlur, Salt, AdjustResolution


class RecDataset(Dataset):
    def __init__(self, config):
        """
        :param config: dataset config, need: textline, input_h, input_w
         mean, std, augmentation, alphabet
        """
        self.textline = config.textline
        self.alphabet = config.alphabet
        self.mean = np.array(config.mean, dtype=np.float32)
        self.std = np.array(config.std, dtype=np.float32)
        self.augmentation = config.augmentation
        self.process = RecDataProcess(config)

        # get alphabet
        with open(self.alphabet, 'r') as file:
            alphabet = ''.join([s.strip('\n') for s in file.readlines()])

        # get converter
        self.converter = StrLabelConverter(alphabet, False)

        # preload all text boxes & trans
        self.data = self.get_data()

        print(f'load {self.__len__()} images.')

    def get_data(self):
        # read textline
        with open(self.textline, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # get labels
        data = list()
        for line in lines:
            img_path, gt_path = line.rstrip('\n').split('\t')
            # read box & trans
            with open(gt_path, 'r', encoding='utf-8-sig') as file:
                boxes_trans = file.readlines()
            # build (img_path, [clockwise coords], trans)
            for item in boxes_trans:
                item = item.rstrip('\n').split(',')
                box, trans = item[: 8], ''.join(item[8: ])
                # drop fuzzy label
                if trans in {'', '*', '###'}:
                    continue
                #  add to labels
                data += [(img_path, box, trans)]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get ori_img_path, box, trans
        ori_img_path, box, trans = self.data[index]

        # convert to label
        label, length = self.converter.encode(trans)

        # read ori_img
        ori_img = cv2.imread(ori_img_path)
        # crop box img
        img = self.process.crop(ori_img, box)
        # do aug
        if self.augmentation:
            img = pil2cv(self.process.aug_img(cv2pil(img)))

        return img, label, length


class RecDataLoader:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.process = RecDataProcess(config)
        self.input_w = config.input_w
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_workers = config.num_workers
        self.iteration = 0
        self.dataiter = None

        # num_data_cache can be modified in config
        self.num_cache = config.num_cache

    def __len__(self):
        return len(self.dataset) // self.batch_size if len(self.dataset) % self.batch_size == 0 \
            else len(self.dataset) // self.batch_size + 1

    def __iter__(self):
        return self

    def init_caches(self):
        # build data cache
        self.caches = dict()
        for i in range(self.num_cache):
            self.caches[i] = list()

    def pack(self, batch_data):
        batch = [[], [], []]
        max_length = max({it[2].item() for it in batch_data})
        # img tensor current shape: B,H,W,C
        all_same_height_images = [self.process.resize_with_specific_height(_[0][0].numpy()) for _ in batch_data]
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # make sure max_img_w is integral multiple of 8
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        for i in range(len(batch_data)):
            _, _label, _length = batch_data[i]
            img = self.process.normalize_img(self.process.width_pad_img(all_same_height_images[i], max_img_w))
            img = img.transpose([2, 0, 1])
            label = torch.zeros([max_length])
            label[:_length.item()] = _label
            batch[0].append(torch.FloatTensor(img))
            batch[1].append(label.to(dtype=torch.int32))
            batch[2].append(torch.IntTensor([max_length]))

        return [torch.stack(batch[0]), torch.stack(batch[1]), torch.cat(batch[2])]

    def build(self):
        self.dataiter = DataLoader(self.dataset, batch_size=1,
                                   shuffle=self.shuffle, num_workers=self.num_workers).__iter__()

    def __next__(self):
        # initialize iteration
        if self.dataiter == None:
            self.init_caches()
            self.build()

        while self.iteration < len(self.dataiter):
            temp = self.dataiter.__next__()
            self.iteration += 1

            cache_id = temp[0].shape[1] // self.input_w
            self.caches[cache_id].append(temp)

            caches_length = [len(self.caches[_]) for _ in self.caches]
            max_cache_length = max(caches_length)

            if max_cache_length == self.batch_size:
                max_length_id = int(np.argmax(caches_length))
                batch_data = self.caches[max_length_id]
                self.caches[max_length_id] = list()

                return self.pack(batch_data)

        else:
            if not hasattr(self, 'rest')  and self.caches is not None:
                # judge whether caches empty
                self.rest = list()
                for k in self.caches:
                    self.rest += self.caches[k]
                self.caches = None

        while hasattr(self, 'rest') and len(self.rest):
            if len(self.rest) > self.batch_size:
                batch_data = self.rest[: self.batch_size]
                self.rest = self.rest[self.batch_size: ]
            else:
                batch_data = self.rest
                self.rest = list()
            return self.pack(batch_data)
        else:
            self.iteration = 0
            self.dataiter = None
            delattr(self, 'rest')
            raise StopIteration


class RecDataProcess:
    def __init__(self, config):
        self.config = config
        self.random_contrast = RandomContrast(probability=0.3)
        self.random_brightness = RandomBrightness(probability=0.3)
        self.random_sharpness = RandomSharpness(probability=0.3)
        self.compress = Compress(probability=0.3)
        self.rotate = Rotate(probability=0.5)
        self.blur = Blur(probability=0.3)
        self.motion_blur = MotionBlur(probability=0.3)
        self.salt = Salt(probability=0.3)
        self.adjust_resolution = AdjustResolution(probability=0.3)
        self.random_line = RandomLine(probability=0.3)
        self.random_contrast.setparam()
        self.random_brightness.setparam()
        self.random_sharpness.setparam()
        self.compress.setparam()
        self.rotate.setparam()
        self.blur.setparam()
        self.motion_blur.setparam()
        self.salt.setparam()
        self.adjust_resolution.setparam()

    def aug_img(self, img):
        img = self.random_contrast.process(img)
        img = self.random_brightness.process(img)
        img = self.random_sharpness.process(img)
        img = self.random_line.process(img)

        if img.size[1] >= 32:
            img = self.compress.process(img)
            img = self.adjust_resolution.process(img)
            img = self.motion_blur.process(img)
            img = self.blur.process(img)
        img = self.rotate.process(img)
        img = self.salt.process(img)
        return img

    def crop(self, img, box):
        # get points, adapt float
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, map(round, map(float, box))))
        # check if rectangle
        if len({x1, y1, x2, y2, x3, y3, x4, y4}) == 4:
            # crop rectangle
            img = img[y1: y4, x1: x2]
        # if polygon, crop minimize circumscribed rectangle
        else:
            x_min, x_max = min((x1, x2, x3, x4)), max((x1, x2, x3, x4))
            y_min, y_max = min((y1, y2, y3, y4)), max((y1, y2, y3, y4))
            img = img[y_min: y_max, x_min: x_max]
        return img

    def resize_with_specific_height(self, _img):
        """
        将图像resize到指定高度
        :param _img:    待resize的图像
        :return:    resize完成的图像
        """
        resize_ratio = self.config.input_h / _img.shape[0]
        return cv2.resize(_img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)

    def normalize_img(self, _img):
        """
        根据配置的均值和标准差进行归一化
        :param _img:    待归一化的图像
        :return:    归一化后的图像
        """
        return (_img.astype(np.float32) / 255 - self.config.mean) / self.config.std

    def width_pad_img(self, _img, _target_width, _pad_value=0):
        """
        将图像进行高度不变，宽度的调整的pad
        :param _img:    待pad的图像
        :param _target_width:   目标宽度
        :param _pad_value:  pad的值
        :return:    pad完成后的图像
        """
        _height, _width, _channels = _img.shape
        to_return_img = np.ones([_height, _target_width, _channels]) * _pad_value
        to_return_img[:_height, :_width, :] = _img
        return to_return_img



if __name__ == '__main__':
    # load test
    import config
    from tqdm import tqdm

    dataset = RecDataset(config)
    dataloader = RecDataLoader(dataset, config)
    # dataloader = DataLoader(dataset, batch_size=1)

    # print(list(dataloader))
    # count = 0
    for i in range(10):
        count = 0
        for batch in tqdm(dataloader):
        # for batch in dataloader:
            count += batch[0].shape[0]
        #     print('-'*100)
        #     print(batch[0].shape, batch[1].shape, batch[2].shape)
        #     print('-'*100)
    #        count += 1
    #     if count == 2:
    #         exit()
    #         print(count)
        print(count)
            # pass
