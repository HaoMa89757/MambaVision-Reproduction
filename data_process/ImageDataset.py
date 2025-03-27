import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        自定义图像数据集类
        Args:
            data_dir (str): 数据集根目录（应包含train/val子目录）
            transform (callable, optional): 图像预处理变换
        """
        self.data_dir = data_dir
        self.transform = transform


        # 获取所有样本路径和标签
        self.samples = []
        classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.JPG', '.JPEG', '.PNG')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图像
        img = Image.open(img_path).convert('RGB')

        # 应用数据增强/预处理
        if self.transform:
            img = self.transform(img)

        return img, label