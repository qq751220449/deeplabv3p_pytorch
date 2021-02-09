import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SegmentationLosses(object):
    """
    定义语义分割所使用的损失函数
    """
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=True, gamma=2, alpha=0.5):
        """
        初始化函数
        :param weight: 类别不平衡时权重参数
        :param size_average: 对图像计算出的损失计算平均值 False True
        :param batch_average: 对小batch的loss进行平均
        :param ignore_index: 计算损失时忽略的类别
        :param cuda: 是否使用CUDA进行计算
        :param gamma: focalloss gamma
        :param alpha: focalloss alpha
        """
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.gamma = gamma
        self.alpha = alpha

    def build_loss(self, loss_type="ce"):
        """
        选择使用的loss的类型
        :param loss_type: 损失函数的类型 ce:交叉熵损失
        :return:
        """
        if loss_type == "ce":
            return self.crossEntroryLoss
        elif loss_type == "focal":
            return self.focalLoss
        elif loss_type == "dice":
            return self.diceLoss
        else:
            raise NotImplementedError

    def crossEntroryLoss(self, logits, targets):
        """
        1.交叉函数内的target必须为long类型，2.交叉函数中target必须为三维（batch, w, h）
        """
        batch_size, channel, image_height, image_width = logits.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction="sum")

        if self.cuda:
            criterion.cuda()

        loss = criterion(logits, targets.long())
        print(loss)
        if self.size_average:
            loss /= (image_height * image_width)
        if self.batch_average:
            loss /= batch_size

        return loss

    def focalLoss(self, logits, targets):
        batch_size, channel, image_height, image_width = logits.size()
        criterion_src = nn.CrossEntropyLoss(weight=None, ignore_index=self.ignore_index, reduction="none")
        criterion_weight = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction="none")
        if self.cuda:
            criterion_src.cuda()
            criterion_weight.cuda()
        logpt = -criterion_src(logits, targets.long())      # 计算出每个像素预测类别的概率
        pt = torch.exp(logpt)                               # 计算出每个像素预测类别的概率
        logweight = -criterion_weight(logits, targets.long())   # 计算带权重的loss输出
        if self.alpha is not None:
            logweight *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logweight
        total_loss = loss.sum()
        if self.size_average:
            total_loss /= (image_height * image_width)
        if self.batch_average:
            total_loss /= batch_size
        return total_loss

    def diceLoss(self, logits, targets):
        if self.cuda:
            targets = self.make_one_hot(targets, logits.size()[1]).cuda()
        else:
            targets = self.make_one_hot(targets, logits.size()[1])
        assert logits.shape == targets.shape, 'predict & target shape do not match'
        total_loss = 0
        predict = F.softmax(logits, dim=1)
        for i in range(targets.shape[1]):
            if i != self.ignore_index:
                dice_loss = self.binaryDiceLoss(predict[:, i], targets[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == targets.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(targets.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
        return total_loss / targets.shape[1]

    def binaryDiceLoss(self, predits, targets, smooth=1, p=2, reduction="mean"):
        assert predits.shape[0] == targets.shape[0], "predict & target batch size don't match"
        predits = predits.contiguous().view(predits.shape[0], -1)
        target = targets.contiguous().view(predits.shape[0], -1)
        num = 2 * torch.sum(torch.mul(predits, target), dim=1) + smooth
        den = torch.sum(predits.pow(p) + target.pow(p), dim=1) + smooth
        loss = 1 - num / den
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(reduction))

    def make_one_hot(self, input, num_classes):
        """Convert class index tensor to one hot encoding tensor.
        one-hot编码实现
        """
        input = input.unsqueeze(1)
        shape = np.array(input.shape)
        shape[1] = num_classes
        shape = tuple(shape)
        result = torch.zeros(shape)
        result = result.scatter_(1, input.cpu(), 1)
        return result


if __name__ == "__main__":
    torch.manual_seed(2)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    weights = torch.tensor([1.0, 37.5, 54.2, 42.5, 35.6, 65.0, 34.6, 12.4, 45.6])
    # loss = SegmentationLosses(weight=weights, cuda=False, batch_average=True, size_average=True).build_loss("ce")
    # loss_focal = SegmentationLosses(weight=weights, cuda=True, batch_average=True, size_average=True, gamma=2, alpha=None).build_loss("focal")
    a = torch.rand(2, 9, 7, 7)
    b = torch.randint(0, 9, (2, 7, 7))
    loss_dicex = SegmentationLosses(weight=weights, cuda=False, batch_average=True, size_average=True, gamma=2, alpha=None).build_loss("dice")
    print(loss_dicex(a, b).item())

