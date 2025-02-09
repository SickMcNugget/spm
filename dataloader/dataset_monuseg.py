import os
from pathlib import Path
import random
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
import cv2

def random_rot_flip(image, label):
    # k--> angle
    # i, j: axis
    k = np.random.randint(0, 4)
    axis = random.sample(range(0, 3), 2)
    image = np.rot90(image, k, axes=(axis[0], axis[1]))  # rot along z axis
    label = np.rot90(label, k, axes=(axis[0], axis[1]))

    # axis = np.random.randint(0, 2)
    # image = np.flip(image, axis=axis)
    # label = np.flip(label, axis=axis)
    flip_id = (
        np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2
        - 1
    )
    image = np.ascontiguousarray(image[:: flip_id[0], :: flip_id[1], :: flip_id[2]])
    label = np.ascontiguousarray(label[:: flip_id[0], :: flip_id[1], :: flip_id[2]])
    return image, label


def random_rotate(image, label, min_value):
    angle = np.random.randint(-10, 10)  # -20--20
    rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    image = ndimage.interpolation.rotate(
        image,
        angle,
        axes=rotate_axes[k],
        reshape=False,
        order=3,
        mode="constant",
        cval=min_value,
    )
    label = ndimage.interpolation.rotate(
        label,
        angle,
        axes=rotate_axes[k],
        reshape=False,
        order=0,
        mode="constant",
        cval=0.0,
    )

    return image, label


def rot_from_y_x(image, label):
    image = np.rot90(image, 2, axes=(1, 2))  # rot along z axis
    label = np.rot90(label, 2, axes=(1, 2))

    return image, label


def flip_xz_yz(image, label):
    flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[:: flip_id[0], :: flip_id[1], :: flip_id[2]])
    label = np.ascontiguousarray(label[:: flip_id[0], :: flip_id[1], :: flip_id[2]])
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        min_value = np.min(image)

        # centercop
        # crop alongside with the ground truth

        index = np.nonzero(label)
        index = np.transpose(index)  # 转置后变成二维矩阵，每一行有三个索引元素，分别对应z,x,y三个方向

        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        # middle point
        z_middle = np.int((z_min + z_max) / 2)
        y_middle = np.int((y_min + y_max) / 2)
        x_middle = np.int((x_min + x_max) / 2)

        Delta_z = np.int((z_max - z_min) / 2) + np.int(self.output_size[0] / 4)  # 3
        Delta_y = np.int((y_max - y_min) / 2) + np.int(self.output_size[1] / 4)  # 8
        Delta_x = np.int((x_max - x_min) / 2) + np.int(self.output_size[2] / 4)  # 8

        # random number of x, y, z
        z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)

        # crop patch
        crop_z_down = z_random - np.int(self.output_size[0] / 2)
        crop_z_up = z_random + np.int(self.output_size[0] / 2)
        crop_y_down = y_random - np.int(self.output_size[1] / 2)
        crop_y_up = y_random + np.int(self.output_size[1] / 2)
        crop_x_down = x_random - np.int(self.output_size[2] / 2)
        crop_x_up = x_random + np.int(self.output_size[2] / 2)

        # padding
        if crop_z_down < 0 or crop_z_up > image.shape[0]:
            delta_z = np.maximum(
                np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0])
            )
            image = np.pad(
                image,
                ((delta_z, delta_z), (0, 0), (0, 0)),
                "constant",
                constant_values=min_value,
            )  # 预处理后填充-1024
            label = np.pad(
                label,
                ((delta_z, delta_z), (0, 0), (0, 0)),
                "constant",
                constant_values=0.0,
            )
            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[1]:
            delta_y = np.maximum(
                np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1])
            )
            image = np.pad(
                image,
                ((0, 0), (delta_y, delta_y), (0, 0)),
                "constant",
                constant_values=min_value,
            )
            label = np.pad(
                label,
                ((0, 0), (delta_y, delta_y), (0, 0)),
                "constant",
                constant_values=0.0,
            )
            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[2]:
            delta_x = np.maximum(
                np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2])
            )
            image = np.pad(
                image,
                ((0, 0), (0, 0), (delta_x, delta_x)),
                "constant",
                constant_values=min_value,
            )
            label = np.pad(
                label,
                ((0, 0), (0, 0), (delta_x, delta_x)),
                "constant",
                constant_values=0.0,
            )
            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x

        label = label[
            crop_z_down:crop_z_up, crop_y_down:crop_y_up, crop_x_down:crop_x_up
        ]
        image = image[
            crop_z_down:crop_z_up, crop_y_down:crop_y_up, crop_x_down:crop_x_up
        ]

        label = np.round(label)

        # data augmentation
        if self.mode == "train":
            if random.random() > 0.5:
                image, label = flip_xz_yz(image, label)
            if random.random() > 0.5:  # elif random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)

        image = torch.from_numpy(image.astype(np.float)).unsqueeze(0).float()
        label = torch.from_numpy(label.astype(np.float32)).float()

        sample = {"image": image, "label": label.long()}
        return sample


class monuseg_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.img_dir = Path(base_dir) / "yolo" / split
        self.mask_dir = Path(base_dir) / "masks" / split
        self.num_classes = 2
        self.split = split
        self.imgs = sorted(self.img_dir.glob("*.png"))
        self.masks = sorted(self.mask_dir.glob("*.png"))

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]))
        mask = cv2.imread(str(self.masks[idx]))

        sample = {"image": img, "label": mask}
        if self.transform:
            sample = self.transform(sample)

        sample["case_name"] = self.masks[idx].stem
        return sample
