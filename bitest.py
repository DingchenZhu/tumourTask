import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from BiDataset import BiCustomDataset


def load_data(data_dir, batch_size=32):
    """
    加载测试数据。

    参数:
    - data_dir: 包含Test文件夹的根目录。
    - batch_size: 每个批次的大小。

    返回:
    - test_loader: 测试数据的DataLoader。
    """

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载测试数据
    test_data_dir = os.path.join(data_dir, 'Test')
    test_dataset = BiCustomDataset(test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def test_model(model, test_loader, device):
    """
    使用测试数据对模型进行测试，并打印预测标签和真实标签。

    参数:
    - model: 已训练的模型。
    - test_loader: 测试数据的DataLoader。
    - device: 设备（'cuda' 或 'cpu'）。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            predicted_labels = predicted.cpu().numpy()
            true_labels = labels.cpu().numpy()

            for i in range(len(predicted_labels)):
                print(f'Predicted: {predicted_labels[i]}, True: {true_labels[i]}')
                if predicted_labels[i] == true_labels[i]:
                    correct += 1
                total += 1
        acc = correct / total
        print(f'Accuracy: {acc}')

if __name__ == '__main__':
# 使用示例
    data_dir = 'D:\TumourTask\data'  # 替换为实际数据路径
    batch_size = 32

    # 加载测试数据
    test_loader = load_data(data_dir, batch_size)

    # 定义模型
    model = models.resnet18(pretrained=True)
    # num_classes = 9
    num_classes = 2  # 根据实际情况定义类别数
    model.fc = nn.Linear(model.fc.in_features, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 加载已经训练好的模型权重
    # model.load_state_dict(torch.load('tumor_classification_model.pth'))

    model.load_state_dict(torch.load('binary_tumor_classification_model.pth'))
    model.eval()

    # 测试模型
    test_model(model, test_loader, device)
