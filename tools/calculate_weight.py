import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.process_label import encode_labels, verify_labels


class WeightDataset(Dataset):

    def __init__(self, csv_file):
        super(WeightDataset, self).__init__()
        self.data = pd.read_csv(csv_file)

        self.labels = self.data["Label_Path"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = cv2.imread(self.labels[item], cv2.IMREAD_GRAYSCALE)
        train_label = encode_labels(label)

        return train_label


def calculate_weigths_labels(csv_file, num_classes):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_label_dataset = WeightDataset(csv_file)
    train_label_dataset_batch = DataLoader(train_label_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)
    z = np.zeros((num_classes,))
    tqdm_batch = tqdm(train_label_dataset_batch)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        sample = np.array(sample)
        # verify_labels(sample[0])
        mask = ((sample >= 0) & (sample < num_classes))
        labels = sample[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        print(count_l)
        z += count_l
    # print(z)
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    # print(ret)
    classes_weights_path = os.path.join(os.getcwd(), "../weights/label_classes_weights_xxx.npy")
    np.save(classes_weights_path, ret)
    return ret


if __name__ == "__main__":
    calculate_weigths_labels(os.path.abspath(os.path.join(os.getcwd(), "../data_list/train.csv")), 9)
    ret = np.load(os.path.join(os.getcwd(), "../weights/label_classes_weights.npy"))
    print(ret)
