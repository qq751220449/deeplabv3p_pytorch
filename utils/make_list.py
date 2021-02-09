import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import config

pd.set_option('display.width', 130)
pd.set_option('display.max_columns', 130)
pd.set_option('display.max_colwidth', 130)

ROOT_DIR = os.getcwd()  # 获取当前文件所在的根目录
# print(os.path.abspath(os.path.dirname(__file__)))   # 作用同上
print("ROOT DIR is ", os.path.abspath(os.path.join(os.getcwd(), "..")))


def dataset_make():
    """
    遍历所有的图片,生成的训练集,校验集
    :return:
    """
    DataPath_Image = os.path.abspath(os.path.join(os.getcwd(), "../dataset/Color_Image"))  # 获取当前数据集所在目录
    # DataPath_Label = os.path.abspath(os.path.join(os.getcwd(), "../dataset/Gray_Label"))  # 获取当前数据集所在目录
    DataListPath = os.path.abspath(os.path.join(os.getcwd(), "../data_list"))  # 数据集列表文件

    Image_Path_List = []
    Label_Path_List = []

    for dirpath, dirnames, filenames in os.walk(DataPath_Image):  # 遍历文件目录
        if len(filenames) > 0:  # 已经遍历到图片所在目录
            for filename in filenames:
                image_path = os.path.abspath(os.path.join(dirpath, filename))
                Image_Path_List.append(image_path)
                # print(image_path)
                dirpath_copy = dirpath
                labename = filename.replace(".jpg", "_bin.png")
                dirpath_copy = dirpath_copy.replace("Color_Image", "Gray_Label")
                dirpath_copy = dirpath_copy.replace("ColorImage_road", "Label_road")
                dirpath_copy = dirpath_copy.replace("ColorImage", "Label")
                if os.path.exists(dirpath_copy):
                    label_path = os.path.abspath(os.path.join(dirpath_copy, labename))
                    Label_Path_List.append(label_path)
                else:
                    print("Label Path is not exist.")

    if len(Image_Path_List) == len(Label_Path_List):
        print("The number of Image is equal to the number of Label.")
    else:
        print("The number of Image is not equal to the number of Label.")

    Image_Label_df = pd.DataFrame({"Image_Path": Image_Path_List, "Label_Path": Label_Path_List})
    print(Image_Label_df.head(10))  # 查看数据集的前10行

    train_dataset, val_dataset = train_test_split(Image_Label_df, random_state=config.dataset_random_state, test_size=config.dataset_test_size)
    Image_Label_df.to_csv(os.path.abspath(os.path.join(DataListPath, "train_val.csv")), index=False)
    train_dataset.to_csv(os.path.abspath(os.path.join(DataListPath, "train.csv")), index=False)
    val_dataset.to_csv(os.path.abspath(os.path.join(DataListPath, "val.csv")), index=False)


if __name__ == "__main__":
    dataset_make()
