import numpy as np


def encode_labels(src_mask):
    encode_mask = np.zeros([src_mask.shape[0], src_mask.shape[1]], dtype=np.uint8)

    # train_id 0
    encode_mask[src_mask == 0] = 0
    encode_mask[src_mask == 249] = 0
    encode_mask[src_mask == 255] = 0

    # train_id 1
    encode_mask[src_mask == 200] = 1
    encode_mask[src_mask == 204] = 1
    encode_mask[src_mask == 213] = 1
    encode_mask[src_mask == 209] = 1
    encode_mask[src_mask == 206] = 1
    encode_mask[src_mask == 207] = 1

    # train_id 2
    encode_mask[src_mask == 201] = 2
    encode_mask[src_mask == 203] = 2
    encode_mask[src_mask == 211] = 2
    encode_mask[src_mask == 208] = 2

    # train_id 3
    encode_mask[src_mask == 216] = 3
    encode_mask[src_mask == 217] = 3
    encode_mask[src_mask == 215] = 3

    # train_id 4
    encode_mask[src_mask == 218] = 4
    encode_mask[src_mask == 219] = 4

    # train_id 5
    encode_mask[src_mask == 210] = 5
    encode_mask[src_mask == 232] = 5

    # train_id 6
    encode_mask[src_mask == 214] = 6

    # train_id 7
    encode_mask[src_mask == 202] = 7
    encode_mask[src_mask == 220] = 7
    encode_mask[src_mask == 221] = 7
    encode_mask[src_mask == 222] = 7
    encode_mask[src_mask == 231] = 7
    encode_mask[src_mask == 224] = 7
    encode_mask[src_mask == 225] = 7
    encode_mask[src_mask == 226] = 7
    encode_mask[src_mask == 230] = 7
    encode_mask[src_mask == 228] = 7
    encode_mask[src_mask == 229] = 7
    encode_mask[src_mask == 233] = 7

    # train_id 8
    encode_mask[src_mask == 205] = 8
    encode_mask[src_mask == 212] = 8
    encode_mask[src_mask == 227] = 8
    encode_mask[src_mask == 223] = 8
    encode_mask[src_mask == 250] = 8

    return encode_mask


def decode_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)
    # 0
    deocde_mask[labels == 0] = 0
    # 1
    deocde_mask[labels == 1] = 200
    # 2
    deocde_mask[labels == 2] = 201
    # 3
    deocde_mask[labels == 3] = 216
    # 4
    deocde_mask[labels == 4] = 218
    # 5
    deocde_mask[labels == 5] = 210
    # 6
    deocde_mask[labels == 6] = 214
    # 7
    deocde_mask[labels == 7] = 202
    # 8
    deocde_mask[labels == 8] = 205

    return deocde_mask


def decode_color_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    # 0
    deocde_mask[labels == 0] = (0, 0, 0)
    # 1
    deocde_mask[labels == 1] = (70, 130, 180)
    # 2
    deocde_mask[labels == 2] = (0, 0, 142)
    # 3
    deocde_mask[labels == 3] = (153, 153, 153)
    # 4
    deocde_mask[labels == 4] = (102, 102, 156)
    # 5
    deocde_mask[labels == 5] = (128, 64, 128)
    # 6
    deocde_mask[labels == 6] = (190, 153, 153)
    # 7
    deocde_mask[labels == 7] = (0, 0, 230)
    # 8
    deocde_mask[labels == 8] = (255, 128, 0)

    return deocde_mask


def verify_labels(labels):
    pixels = [0]
    pixels_numbers = np.zeros(9)
    for i in range(9):
        label_copy = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)
        label_copy[labels == i] = 1
        pixels_numbers[i] = np.sum(label_copy)
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            pixel = labels[x, y]
            if pixel not in pixels:
                pixels.append(pixel)
    print('The Labels Has Value:', pixels, pixels_numbers)


# if __name__ == "__main__":
#     import os
#     import cv2
#     import pandas as pd
#
#     csv_path = DataListPath = os.path.abspath(os.path.join(os.getcwd(), "../data_list/train.csv"))
#     print(csv_path)
#
#     df = pd.read_csv(csv_path)
#     print(len(df["Image_Path"].values))
#     image_path = df["Label_Path"][0]
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     print(image.copy().shape)
#     print(np.copy(image).shape)



    # print(df["Label_Path"][0])
    # image_path = df["Label_Path"][0]
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("src_image", image)
    # cv2.waitKey(0)
    # verify_labels(image)
    # image_encode = encode_labels(image)
    # cv2.imshow("image_encode", image_encode)
    # cv2.waitKey(0)
    # verify_labels(image_encode)
    # image_decode = decode_labels(image_encode)
    # cv2.imshow("image_decode", image_decode)
    # cv2.waitKey(0)
    # verify_labels(image_decode)
    # image_decode_color = decode_color_labels(image_encode)
    # cv2.imshow("image_decode", image_decode_color)
    # cv2.waitKey(0)






