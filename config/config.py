
# 配置文件,设置模型参数

"""数据集生成参数设置"""
dataset_test_size = 0.1                     # 设置训练集：校验集比例 = (1-dataset_test_size):dataset_test_size
dataset_random_state = 33                   # 数据集分割随机数

"""网络模型参数设置"""
backone = "xception"                          # 设置使用backone网络模型 ["resnet", "xception"]
output_stride = 16                          # 设置输出stride [16, 8]
num_classes = 9



"""********学习率参数设置********"""
# 通用参数设置
base_lr = 1e-2          # 设置基础学习率
epoch = 100             # 设置总共训练多少的epoch
lr_mode = "poly"        # 学习率调整模式["step","poly","cos"]
batch_size = 4
warm_up_lr = 0

WEIGHT_DECAY = 1.0e-4

# mode = step 需设置
lr_step = epoch // 5
"""********学习率参数设置********"""


"""训练参数设置"""
use_balanced_weights = True


weights_files = "./weights/label_classes_weights.npy"
loss_type = "ce"

cuda = True
gpu_ids = [0, 1]