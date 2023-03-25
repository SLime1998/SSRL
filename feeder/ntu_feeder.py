import random
import cv2

import numpy
import numpy as np
import pickle, torch
from . import tools


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """
    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            # self.label = np.load(self.label_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy


class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data1 = self._aug(data_numpy)
        data2 = self._aug(data_numpy)


        return [data1, data2], label

    def _aug(self, data_numpy):
        if random.random() < 0.5:
            data_numpy = self.crop_sample_n(data_numpy, 0.8)
            data_numpy = self.real_resize(data_numpy)
        # if self.temperal_padding_ratio > 0:
        #     data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        data_numpy = tools.random_rotate(data_numpy)
        data_numpy = tools.part_reverse_p(data_numpy, self._get_length(data_numpy))

        return data_numpy

    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
            # data_numpy = self.crop_sample(data_numpy, 0.8)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        data_numpy = tools.random_spatial_flip(data_numpy)
        data_numpy = tools.random_rotate(data_numpy)
        data_numpy = tools.random_time_flip(data_numpy)
        data_numpy = tools.gaus_noise(data_numpy)
        data_numpy = tools.gaus_filter(data_numpy)
        data_numpy = tools.axis_mask(data_numpy)
        # data_numpy = tools.part_reverse_p(data_numpy, self._get_length(data_numpy))
        # data_numpy = tools.drop_joint(data_numpy, drop_num=2)

        return data_numpy

    def _strong_aug_2(self, data_numpy):
        data_numpy = self.crop_sample(data_numpy, 0.8)
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
            # data_numpy = self.crop_sample(data_numpy, 0.8)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        data_numpy = tools.random_spatial_flip(data_numpy)
        data_numpy = tools.random_rotate(data_numpy)
        data_numpy = tools.random_time_flip(data_numpy)
        data_numpy = tools.gaus_noise(data_numpy)
        data_numpy = tools.gaus_filter(data_numpy)
        data_numpy = tools.axis_mask(data_numpy)
        data_numpy = tools.part_reverse_p(data_numpy, self._get_length(data_numpy))
        # data_numpy = tools.drop_joint(data_numpy, drop_num=2)

        return data_numpy

    @staticmethod
    def _get_length(data):
        length = (abs(data[:, :, 0, 0]).sum(axis=0) != 0).sum()
        return length

    @staticmethod
    def crop_sample(data, m=0.8):
        #ctvm
        if random.random()>0.5:
            start_frame = random.randint(0, int(data.shape[1] * (1 - m)))
            end_frame = start_frame + int(data.shape[1] * m)

            croped = numpy.zeros_like(data)
            croped[:, start_frame:end_frame] = data[:, start_frame:end_frame]
            return croped
        else:
            return data

    @staticmethod
    def crop_sample_n(data, m=0.8):
        #ctvm
        start_frame = random.randint(0, int(data.shape[1] * (1 - m)))
        end_frame = start_frame + int(data.shape[1] * m)

        croped = data[:, start_frame:end_frame]
        return croped

    @staticmethod
    def real_resize(data_numpy, crop_size=50):
        C, T, V, M = data_numpy.shape
        new_data = np.zeros([C, crop_size, V, M])
        for i in range(M):
            tmp = cv2.resize(data_numpy[:, :, :, i].transpose(
                [1, 2, 0]), (V, crop_size), interpolation=cv2.INTER_LINEAR)
            tmp = tmp.transpose([2, 0, 1])
            new_data[:, :, :, i] = tmp
        return new_data.astype(np.float32)

class Feeder_triple(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            # self.label = np.load(self.label_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data1 = self._strong_aug(data_numpy)
        data2 = self._strong_aug(data_numpy)
        data3 = self._strong_aug_3(data_numpy)


        return [data1, data2, data3], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy

    def _strong_aug(self, data_numpy):
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        data_numpy = tools.random_rotate(data_numpy)


        return data_numpy

    def _strong_aug_3(self, data_numpy):
        pos = random.random()

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        data_numpy = tools.random_rotate(data_numpy)


        if pos < 1/3:
            data_numpy = self.crop_sample_n(data_numpy, 0.8)
            data_numpy = self.real_resize(data_numpy)
        elif pos < 2/3:
            data_numpy = tools.part_zero(data_numpy, self._get_length(data_numpy))
        else:
            data_numpy = self.crop_sample_n(data_numpy, 0.8)
            data_numpy = self.real_resize(data_numpy)
            data_numpy = tools.part_zero(data_numpy, self._get_length(data_numpy))


        return data_numpy

    @staticmethod
    def _get_length(data):
        length = (abs(data[:, :, 0, 0]).sum(axis=0) != 0).sum()
        return length

    @staticmethod
    def real_resize(data_numpy, crop_size=50):
        C, T, V, M = data_numpy.shape
        new_data = np.zeros([C, crop_size, V, M])
        for i in range(M):
            tmp = cv2.resize(data_numpy[:, :, :, i].transpose(
                [1, 2, 0]), (V, crop_size), interpolation=cv2.INTER_LINEAR)
            tmp = tmp.transpose([2, 0, 1])
            new_data[:, :, :, i] = tmp
        return new_data.astype(np.float32)

    @staticmethod
    def crop_sample(data, m=0.8):
        #ctvm
        # if random.random()>0.5:
        start_frame = random.randint(0, int(data.shape[1] * (1 - m)))
        end_frame = start_frame + int(data.shape[1] * m)

        croped = numpy.zeros_like(data)
        croped[:, start_frame:end_frame] = data[:, start_frame:end_frame]
        return croped
        # else:
        #     return data

    @staticmethod
    def crop_sample_n(data, m=0.8):
        #ctvm
        start_frame = random.randint(0, int(data.shape[1] * (1 - m)))
        end_frame = start_frame + int(data.shape[1] * m)

        croped = data[:, start_frame:end_frame]
        return croped



class Feeder_semi(torch.utils.data.Dataset):
    """ Feeder for single inputs """
    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy
