import os
from PIL import Image
from torch.utils.data import Dataset

class BiCustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(class_dir, filename))
                    # 转换为二分类标签
                    if label in [0,2,4,5,6,8]:  # 良性
                        self.labels.append(0)
                    else:  # 恶性
                        self.labels.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label