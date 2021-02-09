import os
import cv2
import numpy as np
import pandas as pd
from utils.process_image import crop_resize_data
from utils.process_label import encode_labels


def train_image_gen(train_list, batch_size=128, image_size=(1024, 384), crop_offset=690):
    # 这里要注意的是opencv读取进来的数据格式是BGR格式,因此需要注意的是如何训练的,最后如何预测
    # Arrange all indexes
    all_batches_index = np.arange(0, len(train_list))
    out_images = []
    out_masks = []
    image_dir = np.array(train_list['Image_Path'])
    label_dir = np.array(train_list['Label_Path'])
    while True:
        # Random shuffle indexes every epoch
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(image_dir[index]):
                image_src = cv2.imread(image_dir[index], cv2.IMREAD_COLOR)
                label_src = cv2.imread(label_dir[index], cv2.IMREAD_GRAYSCALE)
                roi_image, roi_label = crop_resize_data(image_src, label_src, image_size, crop_offset)
                # Encode
                roi_label_encode = encode_labels(roi_label)
                out_images.append(roi_image)
                out_masks.append(roi_label_encode)

                if len(out_images) >= batch_size:   # 表示已经满足batch_size的数据量
                    out_images = np.array(out_images)
                    out_masks = np.array(out_masks)
                    # print(out_images.shape)
                    out_images = out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32)/(255.0/2)-1   # 首先将BGR模式转换为RGB,再进行维度变换,再将数据缩放至[-1,1]
                    out_masks = out_masks.astype(np.int64)
                    # print(out_images.shape, out_masks.shape)
                    yield out_images, out_masks
                    out_images, out_masks = [], []
            else:
                print(image_dir, 'does not exist.')
        break


if __name__ == "__main__":
    df = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), "../data_list/train.csv")))
    for epoch in range(30):
        print("epoch:", epoch)
        for out_images, out_labels in train_image_gen(df):
            print(out_images.shape, out_labels.shape)
