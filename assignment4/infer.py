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

from dataset import create_test_dataloader
from model import ChineseHerbModel

class_names = ['baihe', 'dangshen', 'gouqi', 'huaihua', 'jinyinhua']
num_classes = len(class_names)

def infer():
    # 加载数据集
    test_dataset, test_loader = create_test_dataloader()

    model = ChineseHerbModel(output_dim=num_classes)
    checkpoint_path = ['checkpoints/best_model_batch32_epoch200.pth',
                        'checkpoints/200.pth']
    model.load_state_dict(torch.load(checkpoint_path[1]))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.to(device)

    success = 0
    gt_labels = []
    pred_labels = []
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        success += torch.sum(preds == labels.data)
        gt_labels.extend(class_names[label] for label in labels)
        pred_labels.extend(class_names[pred] for pred in preds)
    succ_rate = success.item() / len(test_dataset)

    print(f"Success: {success.item()}/{len(test_dataset)}, Success Rate: {succ_rate:.2%}")
    print("Ground truth labels:", gt_labels)
    print("Predicted labels:", pred_labels)

if __name__ == "__main__":
    infer()