import os
from utils.data_generator import build_dataset
from config import config
import torch
from models.deeplabv3p import DeepLab
import numpy as np
from tools.calculate_weight import calculate_weigths_labels
from tools.loss import SegmentationLosses
from tools.metrics import Evaluator
from tools.lr_scheduler import LR_Scheduler
from tqdm import tqdm


class Trainer(object):

    def __init__(self):

        self.epoch = config.epoch

        self.trainF = open(("train.txt"), 'w')
        self.testF = open(("val.txt"), 'w')

        # Define DataLoader
        self.train_loader, self.val_loader = build_dataset(config.batch_size)

        # Define Nerwork
        self.model = DeepLab(num_classes=config.num_classes,
                             backbone=config.backone,
                             output_stride=config.output_stride)

        # Define Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.base_lr, weight_decay=config.WEIGHT_DECAY)

        # Define Criterion
        # whether to use class balanced weights
        if config.use_balanced_weights:
            classes_weights_path = os.path.join(os.getcwd(), config.weights_files)
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(os.path.abspath(os.path.join(os.getcwd(), "./data_list/train.csv")), config.num_classes)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=config.cuda).build_loss(loss_type=config.loss_type)
        self.optimizer = optimizer

        # Define Evaluator
        self.evaluator = Evaluator(config.num_classes)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(config.lr_mode, config.base_lr,
                                      config.epoch, len(self.train_loader))

        # Using cuda
        if config.cuda:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=config.gpu_ids)

        self.best_pred = 0.0


    def training(self, epoch):
        # 转换为训练模式
        train_loss = 0.0
        self.model.train()
        dataprocess = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(dataprocess):
            image, target = sample['image'], sample['mask']
            print(image.size(), target.size())
            if config.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)  # 调整学习率

            # optimizer.zero将每个parameter的梯度清0
            self.optimizer.zero_grad()

            # 输出预测的mask
            output = self.model(image)

            # 计算损失
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            dataprocess.set_description_str("epoch:{}".format(epoch))
            dataprocess.set_description_str("step:{}".format(i))
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(loss.item()))
            # dataprocess.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        self.trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, train_loss / num_img_tr))
        self.trainF.flush()

    def validation(self, epoch):

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if config.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.pixel_accuracy()
        Acc_class = self.evaluator.pixel_accurary_class()
        mIoU = self.evaluator.mean_iou()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        self.testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, test_loss / len(self.val_loader)))
        self.testF.write("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            torch.save({'state_dict': self.model.state_dict()},
                       os.path.abspath(os.path.join(os.getcwd(), "./logs/laneNet{}.pth.tar".format(epoch))))


if __name__ == "__main__":
    # torch.manual_seed(config.seed)
    trainer = Trainer()
    for epoch in range(trainer.epoch):
        trainer.training(epoch)
        trainer.validation(epoch)

    print(os.path.abspath(os.path.join(os.getcwd(), "./logs/laneNet{}.pth.tar".format(0))))



