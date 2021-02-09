import numpy as np


class Evaluator(object):

    def __init__(self, num_classes):
        super(Evaluator, self).__init__()

        self.num_classes = num_classes      # 类别数量
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def _generate_confusion_matrix_function1(self, gt_label, pre_label):
        mask = (gt_label >= 0) & (gt_label < self.num_classes)
        label = self.num_classes * np.array(gt_label)[mask].astype(np.int) + np.array(pre_label)[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def _generate_confusion_matrix_function2(self, gt_label, pre_label):
        gt_label_array, pre_label_array = np.copy(gt_label), np.copy(pre_label)
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            confusion_matrix_class = np.bincount(pre_label_array[gt_label_array == i], minlength=self.num_classes)
            confusion_matrix[i] = confusion_matrix_class
        return confusion_matrix

    def add_batch(self, gt_label, pre_label):
        assert gt_label.shape == pre_label.shape
        self.confusion_matrix += self._generate_confusion_matrix_function2(gt_label, pre_label)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accurary_class(self):
        acc_matrix = np.zeros((self.num_classes,), dtype=np.float32)
        diag = np.diag(self.confusion_matrix)
        sum_class = self.confusion_matrix.sum(axis=1)
        for i in range(self.num_classes):
            if sum_class[i] == 0:
                acc_matrix[i] = 0.0
            else:
                acc_matrix[i] = diag[i] / sum_class[i]
        return acc_matrix

    def mean_iou(self):
        miou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix))
        miou = np.nanmean(miou)
        return miou

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


if __name__ == "__main__":
    import os
    import cv2
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from utils.process_label import encode_labels

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


    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_label_dataset = WeightDataset(os.path.abspath(os.path.join(os.getcwd(), "../data_list/train.csv")))
    train_label_dataset_batch = DataLoader(train_label_dataset, batch_size=2, shuffle=False, drop_last=False, **kwargs)

    eval = Evaluator(9)
    for xxx in train_label_dataset_batch:
        print(xxx.size())
        eval.add_batch(xxx[0], xxx[0])
        print(eval.pixel_accuracy())
        print(eval.pixel_accurary_class())
        print(eval.confusion_matrix)
        # print(eval.mean_iou())
        print(eval.Frequency_Weighted_Intersection_over_Union())
        break
