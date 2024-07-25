import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Dataset import CustomDataset
from transformer import *
from data import *
import timm

def train_model(model, train_loader, val_loader, device, num_epochs=20, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        val_accuracies.append(accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

        scheduler.step()
    return train_losses, val_accuracies


def plot_training(train_losses, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss and Validation Accuracy")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_dir = 'D:\TumourTask\data'  # 替换为实际数据路径
    batch_size = 32
    num_classes = 9
    num_epochs = 20
    learning_rate = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_data(data_dir, batch_size=batch_size)

    model = create_model(num_classes)
    model = model.to(device)

    train_losses, val_accuracies = train_model(model, train_loader, val_loader, device, num_epochs, learning_rate)
    plot_training(train_losses, val_accuracies)

    # 保存模型
    torch.save(model.state_dict(), 'disease_classification_vit.pth')
