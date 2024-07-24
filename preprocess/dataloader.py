import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(data_dir, batch_size=32):
    """
    加载和预处理数据。

    参数:
    - data_dir: 包含九个疾病文件夹的根目录。
    - batch_size: 每个批次的大小。
    - train_split: 训练集所占的比例。

    返回:
    - train_loader: 训练数据的DataLoader。
    - val_loader: 验证数据的DataLoader。
    """

    # 定义数据预处理和增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # data_set = datasets.ImageFolder(root=data_dir, transform=transform)
    train_path = os.path.join(data_dir, 'Train')
    test_path = os.path.join(data_dir, 'Test')
    train_data = datasets.ImageFolder(train_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


data_dir = '../data'  # 替换为实际数据路径
batch_size = 32

train_loader = load_data(data_dir, batch_size=batch_size)
test_loader = load_data(data_dir, batch_size=batch_size)
