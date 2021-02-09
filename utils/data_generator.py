import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.process_image import LaneDataset, ImageAug, DeformAug
from utils.process_image import ScaleAug, CutOut, ToTensor


def build_dataset(batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    # kwargs = {'num_workers': 4, 'pin_memory': True}
    training_dataset = LaneDataset(os.path.abspath(os.path.join(os.getcwd(), "./data_list/train.csv")),
                                   transform=transforms.Compose(
                                       [ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5), ToTensor()]))
    testing_dataset = LaneDataset(os.path.abspath(os.path.join(os.getcwd(), "./data_list/val.csv")),
                                   transform=transforms.Compose([ToTensor()]))
    training_data_batch = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    testing_data_batch = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    return training_data_batch, testing_data_batch



if __name__ == "__main__":
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    # print(os.path.abspath(os.path.join(os.getcwd(), "../data_list/train.csv")))


    training_dataset = LaneDataset(os.path.abspath(os.path.join(os.getcwd(), "../data_list/train.csv")),
                                   transform=transforms.Compose(
                                       [ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5), ToTensor()]))


    # training_dataset = LaneDataset(os.path.abspath(os.path.join(os.getcwd(), "../data_list/train.csv")),
    #                                transform=None)


    # DeformAug(),
    # 真正开始处理数据
    training_data_batch = DataLoader(training_dataset, batch_size=32, shuffle=True, drop_last=True, **kwargs)
    for batch_item in training_data_batch:
        image, mask = batch_item['image'], batch_item['mask']  # 得到的就是经过数据处理的
        # image, mask = batch_item[0], batch_item[1]  # 得到的就是经过数据处理的
        if torch.cuda.is_available():
            image, mask = image.cuda(), mask.cuda()

        # 如果有模型的话，就是讲数据加载进模型开始训练了
        #  prediction = model(image)
        # loss = f (prediction,mask)

        print(image.size())
        print(mask.size())
