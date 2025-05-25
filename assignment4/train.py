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
from tqdm import tqdm
import copy

from dataset import create_train_dataloader
from model import ChineseHerbModel

class_names = ['baihe', 'dangshen', 'gouqi', 'huaihua', 'jinyinhua']
num_classes = len(class_names)

def batch_count_labels(dataset, num_classes, batch_size=1024):
    counts = torch.zeros(num_classes, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for _, labels in loader:
        counts += torch.bincount(labels, minlength=num_classes)
    return counts

def train():
    torch.manual_seed(42)
    # 训练参数
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-3
    checkpoint_every = 25

    # 加载数据集
    train_dataset, val_dataset, train_loader, val_loader = create_train_dataloader(batch_size=batch_size)

    # 输出标签占比
    print('-' * 50)
    print("Train dataset class distribution:")
    train_counts = batch_count_labels(train_dataset, num_classes, batch_size)
    for i, count in enumerate(train_counts):
        print(f"Class {class_names[i]}: {count.item()} samples, "
              f"Percentage: {count.item() / len(train_dataset) * 100:.2f}%")
    print('-' * 50)
    print("Validation dataset class distribution:")
    val_counts = batch_count_labels(val_dataset, num_classes, batch_size)
    for i, count in enumerate(val_counts):
        print(f"Class {class_names[i]}: {count.item()} samples, "
              f"Percentage: {count.item() / len(val_dataset) * 100:.2f}%")

    model = ChineseHerbModel(output_dim=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 记录训练过程
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_acc = 0.0
    best_model = None

    # os.mkdir('checkpoints')

    for epoch in range(num_epochs):
        print('-' * 50)
        cur_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{num_epochs}, Learning rate: {cur_lr}')
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.cpu().numpy())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 测试阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc.cpu().numpy())
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        scheduler.step()

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_state_dict = model.state_dict()

        if (epoch + 1) % checkpoint_every == 0:
            torch.save(model.state_dict(), f'checkpoints/{epoch+1}.pth')
            print(f'Model saved at epoch {epoch+1}')

    torch.save(best_model_state_dict, f'checkpoints/best_model_batch{batch_size}_epoch{num_epochs}.pth')
    print(f'Best Test Acc: {best_acc:.4f}')

    visualize_results(train_loss_history, train_acc_history, val_loss_history, val_acc_history)

def visualize_results(train_loss_history, train_acc_history, val_loss_history, val_acc_history):
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axhline(y=0.9, color='gray', linestyle='--', label='0.9 Accuracy', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    train()