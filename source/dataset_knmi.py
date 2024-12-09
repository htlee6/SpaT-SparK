from random import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
# from torchvision.transforms import v2 as transforms

from datetime import datetime, timedelta


# This contains the whole training dataset (3 years)
class precipitation_maps_h5(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_h5, self).__init__()
        self.file_name = in_file
        self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if train else "test"]['images']
        self.n_images, self.nx, self.ny = self.dataset.shape
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        img = torch.tensor(self.dataset[index], dtype=torch.float32).unsqueeze(0)
        img_next_ = torch.tensor(self.dataset[index+1], dtype=torch.float32).unsqueeze(0)
        # add transforms
        if self.transform is not None:
            img = self.transform(img)
            img_next_ = self.transform(img_next_)
        input_img = img
        target_img = img_next_

        return input_img, target_img

    def __len__(self):
        return self.n_images - 1


class precipitation_maps_oversampled_h5(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5, self).__init__()

        self.file_name = in_file
        self.samples = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape[0]

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.train = train
        self.transform = transform
        self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"]['images']

    def __getitem__(self, index):

        imgs = np.array(self.dataset[index], dtype="float32")
        imgs = torch.from_numpy(imgs)

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[self.num_input:self.num_input + self.num_output]

        return input_img, target_img

    def __len__(self):
        return self.samples

    def __slice__(self, start, end):
        return self.dataset[start:end]


class precipitation_maps_oversampled_TCHW(precipitation_maps_oversampled_h5):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_TCHW, self).__init__(in_file, num_input_images, num_output_images, train, transform)

    def __getitem__(self, index):
        THWimg = super(precipitation_maps_oversampled_TCHW, self).__getitem__(index)
        THWimg_inp, THWimg_oup = THWimg

        return THWimg_inp.unsqueeze(1), THWimg_oup.unsqueeze(1)

    def __slice__(self, start, end):
        BTHWimg = super(precipitation_maps_oversampled_TCHW, self).__slice__(start, end)
        return BTHWimg.unsqueeze(1)


class PrecipitationDataset3D(precipitation_maps_oversampled_h5):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(PrecipitationDataset3D, self).__init__(in_file, num_input_images, num_output_images, train, transform)

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"][
                'images']
        imgs = np.array(self.dataset[index], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = torch.Tensor(imgs[:self.num_input])
        target_img = torch.Tensor(imgs[-1])

        return input_img.unsqueeze(0), target_img.unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return self.samples



class precipitation_maps_oversampled_h5_2years(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5_2years, self).__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        if index >= self.samples - 1911:
            raise IndexError("Index out of range of available data")

        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"][
                'images']

        # Shift the index by 3822
        shifted_index = index + 1911
        imgs = np.array(self.dataset[shifted_index], dtype="float32")

        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return self.samples - 1911


class precipitation_maps_oversampled_h5_1years(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5_1years, self).__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        if index >= 1911:
            raise IndexError("Index out of range of available data")

        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"][
                'images']

        # Shift the index by 3822
        shifted_index = index + 3822
        imgs = np.array(self.dataset[shifted_index], dtype="float32")

        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return 1911


class precipitation_maps_oversampled_h5_pre(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5_pre, self).__init__()

        self.file_name = in_file
        self.samples, self.seq_len, self.w, self.h = h5py.File(self.file_name, 'r')["train" if train else "test"][
            'images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"][
                'images']
        imgs = np.array(self.dataset[index], dtype="float32")

        # add transforms to each image separately
        transformed_images = []

        if self.transform is not None:
            for img in imgs[:12]:
                # min-max normalize
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
                img = Image.fromarray(img.astype(np.uint8), mode='L')

                tensor1, tensor2 = self.transform(img)
                stacked_tensors = torch.stack([tensor1, tensor2], dim=0)
                transformed_images.append(stacked_tensors.numpy())

            # stack the 12 channels together
            # input_img = torch.stack(transformed_images[:self.num_input], dim=0)
            input_img = (np.stack([x[0] for x in transformed_images], axis=0).squeeze(1),
                         np.stack([x[1] for x in transformed_images], axis=0).squeeze(1))
            target_img = transformed_images[
                -1]  # target_img can be arbitrary value because we don't actually use the value
        else:
            raise NotImplementedError("Transform is required for next steps")

        return input_img, target_img

    def __len__(self):
        return self.dataset.shape[0]


class precipitation_maps_oversampled_h5_timecontrastive(precipitation_maps_oversampled_h5_pre):
    def __init__(self, in_file, num_input_images, num_output_images, time_window, train=None, transform=None):
        super(precipitation_maps_oversampled_h5_timecontrastive, self).__init__(in_file, num_input_images,
                                                                                num_output_images, train, transform)
        trainset_img = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)['train']['images']
        testset_img = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)['test']['images']
        trainset_timestamp = np.array(h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)['train']['timestamps'])
        testset_timestamp = np.array(h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)['test']['timestamps'])

        datetime_conversion = np.vectorize(
            lambda s: datetime.strptime(s.decode('utf-8').replace('-', ' ').replace(';', ' '), '%d %b %Y %H:%M:%S.%f'))

        for i in range(trainset_timestamp.shape[0]):
            for j in range(trainset_timestamp.shape[1]):
                try:
                    trainset_timestamp[i, j] = datetime_conversion(trainset_timestamp[i, j])
                except:
                    trainset_timestamp[i, j] = trainset_timestamp[i, j - 1] + timedelta(minutes=5)

        for i in range(testset_timestamp.shape[0]):
            for j in range(testset_timestamp.shape[1]):
                try:
                    testset_timestamp[i, j] = datetime_conversion(testset_timestamp[i, j])
                except:
                    testset_timestamp[i, j] = testset_timestamp[i, j - 1] + timedelta(minutes=5)

        self.dataset = np.concatenate((trainset_img[:, :, :, :], testset_img[:, :, :, :]), axis=0)
        self.dataset_timestamp = np.concatenate((trainset_timestamp[:, :, :], testset_timestamp[:, :, :]), axis=0)
        self.time_window = time_window
        self.positive_sample_id = list()
        self.negative_sample_id = list()

        def sample_within_timewindow(target_cen_idx, time_window: int):

            '''
            time_window: in minutes

                """
                if target_cen_idx - time_window >= 0 and target_cen_idx + time_window < self.__len__():
                    img_time_idx = np.random.randint(target_cen_idx - time_window, target_cen_idx + time_window)
                elif target_cen_idx - time_window < 0:
                    img_time_idx = np.random.randint(0, target_cen_idx + time_window)
                else:
                    img_time_idx = np.random.randint(target_cen_idx - time_window, self.__len__())
                """
            '''

            possible_indices = np.arange(max(0, target_cen_idx - time_window),
                                         min(self.__len__(), target_cen_idx + time_window))
            possible_indices = np.delete(possible_indices, np.where(possible_indices == target_cen_idx))
            for pid in possible_indices:
                # compare in minutes
                if abs(((self.dataset_timestamp[pid, 0, 0]) - (self.dataset_timestamp[target_cen_idx, 0, 0])).total_seconds()) / 60. > float(time_window*5.):
                    possible_indices = np.delete(possible_indices, np.where(possible_indices == pid))

            if len(possible_indices) > 0:
                pass
            else:
                if target_cen_idx == 0:
                    possible_indices = np.array([1])
                elif target_cen_idx == self.__len__() - 1:
                    possible_indices = np.array([self.__len__() - 2])
                else:
                    possible_indices = np.array([target_cen_idx - 1, target_cen_idx + 1])

            return possible_indices

        def sample_outside_timewindow(target_cen_idx, time_window: int):
            """
                if target_cen_idx - time_window > 0 and target_cen_idx + time_window < self.__len__():
                    img_time_idx = np.random.choice([
                        np.random.randint(max(0, target_cen_idx-2*time_window), target_cen_idx - time_window),
                        np.random.randint(target_cen_idx + time_window, min(self.__len__(), target_cen_idx + 2*time_window))
                ])
                elif target_cen_idx - time_window <= 0:
                    img_time_idx = np.random.randint(target_cen_idx + time_window, min(self.__len__(), target_cen_idx + 2*time_window))
                else:
                    img_time_idx = np.random.randint(max(0, target_cen_idx-2*time_window), target_cen_idx - time_window)
            """

            possible_indices = np.arange(max(0, target_cen_idx - 2 * time_window),
                                         max(0, target_cen_idx - time_window))  # left side
            possible_indices = np.append(possible_indices, np.arange(min(self.__len__(), target_cen_idx + time_window),
                                                                     min(self.__len__(),
                                                                         target_cen_idx + 2 * time_window)))
            possible_indices = np.delete(possible_indices, np.where(possible_indices == target_cen_idx))
            for pid in possible_indices:
                if abs(((self.dataset_timestamp[pid, 0, 0]) - (self.dataset_timestamp[target_cen_idx, 0, 0])).total_seconds()) / 60. > float(2 * time_window * 5.):
                    # remove pid from possible_indices
                    possible_indices = np.delete(possible_indices, np.where(possible_indices == pid))
            if len(possible_indices) > 0:
                pass
            else:
                if target_cen_idx == 0:
                    possible_indices = np.array([1])
                elif target_cen_idx == self.__len__() - 1:
                    possible_indices = np.array([self.__len__() - 2])
                else:
                    possible_indices = np.array([target_cen_idx - 1, target_cen_idx + 1])
            return possible_indices

        for i in range(self.__len__()):
            self.positive_sample_id.append(sample_within_timewindow(i, self.time_window))
            self.negative_sample_id.append(sample_outside_timewindow(i, self.time_window))

    def __getitem__(self, index):
        def normalize(img_in):
            img_out = (img_in - img_in.min()) / (img_in.max() - img_in.min()) * 255.0
            img_out = Image.fromarray(img_out.astype(np.uint8), mode='L')
            return img_out

        # load the file here (load as singleton)

        prob = random()
        if prob < 0.5:
            sample_time_idx = np.random.choice(self.positive_sample_id[index])
            y = torch.Tensor([1.]).to(torch.float32)
        else:
            sample_time_idx = np.random.choice(self.negative_sample_id[index])
            y = torch.Tensor([0.]).to(torch.float32)

        possible_time_idx = list()
        possible_time_idx.append(index-1) if index-1 >= 0 else None
        possible_time_idx.append(index+1) if index+1 < self.__len__() else None
        # random choice from list
        sample_time_idx = np.random.choice(possible_time_idx)

        imgs = np.array(self.dataset[index], dtype="float32")
        imgs_time = np.array(self.dataset[sample_time_idx], dtype="float32")

        # add transforms to each image separately
        transformed_images = []
        ys = []

        if self.transform is not None:
            img, img_time = torch.Tensor(imgs[:self.num_input]), torch.Tensor(imgs_time[:self.num_input])
            x, x_time, x_aug, y = self.transform(img, img_time, index, sample_time_idx)
            # for img, img_time in zip(imgs[:self.num_input], imgs_time[:self.num_input]):
            #     min-max normalize
            #     img, img_time = normalize(img), normalize(img_time)
            #
            #    x, x_time, x_aug, y = self.transform(img, img_time, index, sample_time_idx)
            #    stacked_tensors = torch.stack([x, x_time, x_aug], dim=0)
            #    transformed_images.append(stacked_tensors.numpy())

            # stack the 12 channels together
            # input_img = torch.stack(transformed_images[:self.num_input], dim=0)
            # input_img = (
            #     torch.stack([torch.Tensor(x[0]) for x in transformed_images], dim=0).squeeze(1),  # x
            #     torch.stack([torch.Tensor(x[1]) for x in transformed_images], dim=0).squeeze(1),  # x_time
            #     torch.stack([torch.Tensor(x[2]) for x in transformed_images], dim=0).squeeze(1),  # x_aug (moco)
            # )
            input_img = (x, x_time, x_aug)
            # target_img = transformed_images[-1]  # target_img can be arbitrary value because we don't actually use the value
            y = torch.Tensor([y])
        else:
            raise NotImplementedError("Transform is required for next steps")

        return input_img, y

    def __len__(self):
        return self.dataset.shape[0]


class precipitation_maps_classification_h5(Dataset):
    def __init__(self, in_file, num_input_images, img_to_predict, train=True, transform=None):
        super(precipitation_maps_classification_h5, self).__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.img_to_predict = img_to_predict
        self.sequence_length = num_input_images + img_to_predict
        self.bins = np.array([0.0, 0.5, 1, 2, 5, 10, 30])

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images - (num_input_images + img_to_predict)
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"][
                'images']
        imgs = np.array(self.dataset[index:index + self.sequence_length], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        # put the img in buckets
        target_img = imgs[-1]
        # target_img is normalized by dividing through the highest value of the training set. We reverse this.
        # Then target_img is in mm/5min. The bins have the unit mm/hour. Therefore we multiply the img by 12
        buckets = np.digitize(target_img * 47.83 * 12, self.bins, right=True)

        return input_img, buckets

    def __len__(self):
        return self.size_dataset


if __name__ == '__main__':
    pass