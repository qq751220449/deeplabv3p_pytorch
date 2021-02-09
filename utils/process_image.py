import cv2
import torch
import random
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from utils.process_label import encode_labels, decode_labels, decode_color_labels, verify_labels


def crop_resize_data(image, label=None, image_size=(1024, 384), offset=690):
    """
    对图像进行裁切和缩放
    :param image:输入待变换的图像
    :param label:输入带变换的标签
    :param image_size:图像需要缩放的大小
    :param offset: 图像偏移量
    :return:返回变换后的图像和标签
    """
    # image_height, image_width = image.shape[0], image.shape[1]
    if label is not None:
        roi_image = image[offset:, :]
        roi_label = label[offset:, :]

        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_AREA)  # 可以采用双线性插值
        train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST)  # 采用最近邻插值保证图片的label不会发生变化

        return train_image, train_label

    else:
        roi_image = image[offset:, :]
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_AREA)  # 可以采用双线性插值
        return train_image


class LaneDataset(Dataset):
    """
    车道线处理数据集
    """

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        # 将数据加载进来
        self.images = self.data["Image_Path"]  # 保存image路径
        self.labels = self.data["Label_Path"]  # 保存mask路径

        self.number = len(self.images)  # 统计训练数据量
        self.transform = transform

    def __len__(self):
        """
        计算数据集的大小,用于调整epoch,batch_size
        :return:返回数据集大小
        """
        return len(self.images)

    def __getitem__(self, item):
        # 在该函数内部对图像做处理,返回一个单个样本

        image = cv2.imread(self.images[item], cv2.IMREAD_COLOR)
        # print(self.images[item])
        label = cv2.imread(self.labels[item], cv2.IMREAD_GRAYSCALE)  # 读取灰度图

        train_image, train_label = crop_resize_data(image, label)
        # 标签encode
        train_label = encode_labels(train_label)
        sample = [np.copy(train_image), np.copy(train_label)]
        # verify_labels(train_label)
        if self.transform is not None:
            # 需进行数据增强
            sample = self.transform(sample)

        return sample


# 图像增强,封装为类,可以使用transform.complice进行diaoyong
class ImageAug(object):
    def __call__(self, sample):
        image, label = sample  # 传递过来的是字典

        # 生成概率,对数据进行随机的增强
        if np.random.uniform(0, 1) > 0.5:  # 设定0-1之间的随机数,p设置为0.5
            seq = iaa.Sequential([iaa.OneOf(
                [iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),  # 增加高斯噪声
                 iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                 iaa.GaussianBlur(sigma=(0, 1.0))
                 ])])

            image = seq.augment_image(image)  # 对图像进行增强

        return image, label


# deformation augmentation
class DeformAug(object):
    """
    # 图像进行随机裁切
    """

    def __call__(self, sample):
        image, mask = sample
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1), keep_size=True)])
        seg_to = seq.to_deterministic()
        image = seg_to.augment_image(image)
        mask = seg_to.augment_image(mask)
        return image, mask


class ScaleAug(object):
    def __call__(self, sample):
        image, mask = sample
        scale = random.uniform(0.7, 1.5)
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()

        # 对我们的image和mask进行缩放处理
        aug_image = cv2.resize(aug_image, (int(scale * w), int(scale * h)), cv2.INTER_LINEAR)
        aug_mask = cv2.resize(aug_mask, (int(scale * w), int(scale * h)), cv2.INTER_NEAREST)

        if (scale < 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            aug_image = np.pad(aug_image, pad_list, mode="constant")
            aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant")
        if (scale > 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_crop = int((new_h - h) / 2)
            pre_w_crop = int((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            aug_image = aug_image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
        return aug_image, aug_mask


class CutOut(object):
    """
    从图像中某一个部分挖除一个小块
    """

    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, sample):
        image, mask = sample

        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        h, w = image.shape[:2]

        # 找到mask的中心位置
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)

        xmin, ymin = cx - mask_size_half, cy - mask_size_half  # 左上角的点
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size  # 右下角的点

        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)

        if np.random.uniform(0, 1) < self.p:
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
            mask[ymin:ymax, xmin:xmax] = 0
        # mask_decode = decode_color_labels(mask)
        # cv2.imshow("image", image)
        # cv2.imshow("label", mask_decode)
        # cv2.waitKey(0)
        # verify_labels(mask)
        return image, mask


class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        mask = mask.astype(np.long)
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy())}


def expand_resize_data(prediction=None, submission_size=(3384, 1710), offset=690):
    pred_mask = decode_labels(prediction)
    expand_mask = cv2.resize(pred_mask, (submission_size[0], submission_size[1] - offset),
                             interpolation=cv2.INTER_NEAREST)
    submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
    submission_mask[offset:, :] = expand_mask
    return submission_mask


def expand_resize_color_data(prediction=None, submission_size=(3384, 1710), offset=690):
    color_pred_mask = decode_color_labels(prediction)
    color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
    color_expand_mask = cv2.resize(color_pred_mask, (submission_size[0], submission_size[1] - offset),
                                   interpolation=cv2.INTER_NEAREST)
    color_submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')
    color_submission_mask[offset:, :, :] = color_expand_mask
    return color_submission_mask
