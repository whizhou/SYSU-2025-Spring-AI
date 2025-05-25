import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models

# 定义中药类别
class_names = ['baihe', 'dangshen', 'gouqi', 'huaihua', 'jinyinhua']
num_classes = len(class_names)

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Resize(256),  # 先缩放到较大尺寸
    transforms.RandomResizedCrop(224),  # 随机裁剪并缩放回224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
])

test_transform = transforms.Compose([
    transforms.Resize(256),  # 缩放
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ChineseHerbDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.images = []
        self.labels = []
        
        if train:
            # 遍历每个类别文件夹
            for label, class_name in enumerate(class_names):
                class_dir = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)
        else:
            for img_name in os.listdir(root_dir):
                img_path = os.path.join(root_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_names.index(img_name.split('0')[0]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # 确保转换为RGB
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloader():
    # 创建数据集和数据加载器
    train_dataset = ChineseHerbDataset('data/train', transform=train_transform, train=True)
    test_dataset = ChineseHerbDataset('data/test', transform=test_transform, train=False)

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataset, test_dataset, train_loader, test_loader

def create_train_dataloader(batch_size=32):
    # 创建数据集和数据加载器
    dataset = ChineseHerbDataset('data/train', transform=train_transform, train=True)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataset, val_dataset, train_loader, val_loader

def create_test_dataloader(batch_size=32):
    # 创建测试集数据加载器
    test_dataset = ChineseHerbDataset('data/test', transform=test_transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_dataset, test_loader

if __name__ == "__main__":
    # 创建数据加载器
    train_dataset, test_dataset, train_loader, test_loader  = create_train_dataloader()

    # 可视化一些样本
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 反标准化函数
    def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)  # 逆操作: (x * std) + mean
        return tensor.clamp_(0, 1)  # 确保值在[0,1]范围内

    # 显示图像
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        # 反标准化并调整维度顺序
        img = denormalize(images[i].clone()).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    # plt.show()
    plt.savefig('sample_images.png')
