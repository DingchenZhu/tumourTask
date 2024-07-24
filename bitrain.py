import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from PIL import Image
from BiDataset import BiCustomDataset

# num_classes = 9
num_classes = 2
batch_size = 32
num_epochs = 20
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data_dir, batch_size=32,train_split=0.8):
    """
    参数:
    - data_dir: 包含Train Test两个疾病文件夹的根目录。
    - batch_size: 每个批次的大小。
    """


    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载训练数据
    train_data_dir = os.path.join(data_dir, 'Train')
    train_dataset = BiCustomDataset(train_data_dir, transform=transform)

    # 计算训练集和验证集的大小
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # 划分训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 创建训练和验证DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader

# if __name__ == '__main__':
#     path = 'D:\TumourTask\data'
#     train_data, test_data = load_data(path)
#     itr = iter(train_data)
#     images, labels = next(itr)
#     images = images
#     labels = labels.numpy()
#     print(labels)

if __name__ == '__main__':
    data_dir = 'D:\TumourTask\data'  # 替换为实际数据路径
    batch_size = 32

    train_loader , val_loader = load_data(data_dir, batch_size=batch_size)
    # test_loader = load_data(data_dir, batch_size=batch_size)


    # 定义模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    #记录
    train_losses, val_accuracies = [], []

    # 定义损失函数和优化器
    #################################################################
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                print(f"Error during backward pass: {e}")
            # loss.backward()
            # optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                # _, predicted = torch.max(outputs, 1)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        val_accuracies.append(accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

    # 保存模型
    # torch.save(model.state_dict(), 'tumor_classification_model.pth')

    torch.save(model.state_dict(), 'binary_tumor_classification_model.pth')

    # 可视化训练过程
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()

    # 绘制验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Epochs')
    plt.legend()

    plt.show()