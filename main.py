'''
Training and Testing on CIFAR10/CIFAR100 with CNN-based models and ViT-based models!
Take inspiration from https://github.com/kentaroy47/vision-transformers-cifar10
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import csv

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import timm
import timm.data

from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

from models.resnet_cifar import ResNet
from data_loader import get_data_loaders, plot_images
from trainer import train_one_epoch, evaluate
from utils import save_checkpoint, load_checkpoint, plot_history, plot_accuracy

# parsers
parser = argparse.ArgumentParser(description='CIFAR10/100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--milestone', nargs='+', default=[60, 120, 160], type=int, help='milestones for MultiStepLR')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for MultiStepLR')
parser.add_argument('--optimizer', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='vit')
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patch', default=4, type=int)
parser.add_argument('--datadir', default='data/cifar10', type=str, help='dataset to use (cifar10 or cifar100)')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10 or cifar100)')

args = parser.parse_args()

print(f"Arguments:\n{args}\n")
print("Start running ...\n")

# take in args
image_size = int(args.image_size)
datadir = args.datadir
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
milestone = args.milestone
gamma = args.gamma

if args.model == "vit_timm":
    print("For fine-tuning ViT, these default hyperparameters will be used:")
    lr = 1e-4
    weight_decay = 5e-4
    epochs = 12
    batch_size = 64
    print(f"Learning Rate for ViT_Timm: {lr}\n")
    print(f"Weight Decay for ViT_Timm: {weight_decay}\n")
    print(f"Epochs for ViT_Timm: {epochs}\n")
    print(f"Batch Size for ViT_Timm: {batch_size}\n")

# Paths
results_path = f'results/Resnet/{args.dataset}' if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"] \
                                else f'results/{args.model}/{args.dataset}'
os.makedirs(results_path, exist_ok=True)

checkpoint_path = os.path.join(results_path, f'best_model_{args.model}.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

best_test_acc = 0.0


# History (for plotting)
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
    'lr': []
}

# Preparing dataset
print('==> Preparing data..')
if args.model=="vit_timm":
    size = 224
else:
    size = image_size


# Set up normalization based on the dataset
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 10
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
else:
    raise ValueError("Dataset must be either 'cifar10' or 'cifar100'")

if args.model == "vit_timm":
    # ViT pretrained on ImageNet — use ImageNet stats and timm's auto-config
    _tmp = timm.create_model("vit_base_patch16_224", pretrained=False)
    data_config = timm.data.resolve_model_data_config(_tmp)
    del _tmp

    train_transform = timm.data.create_transform(**data_config, is_training=True)
    test_transform  = timm.data.create_transform(**data_config, is_training=False)

elif args.model in ["resnet20", "resnet32", "resnet44", "resnet56"]:
    size = image_size

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
else:
    raise ValueError(f"'{args.model}' is not a valid model")


train_loader, test_loader = get_data_loaders(datadir,
                                             batch_size,
                                             train_transform,
                                             test_transform,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)

# # Set up class names based on the dataset
# if args.dataset == 'cifar10':
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# else:
#     # CIFAR100 has 100 classes, so we don't list them all here
#     classes = None

# Model factory..
print('==> Building model..')
if args.model=='resnet20':
    model = ResNet(n=3, shortcuts=True).to(device)
    print(f"Using ResNet-20")
elif args.model=='resnet32':
    model = ResNet(n=5, shortcuts=True).to(device)
    print(f"Using ResNet-32")
elif args.model=='resnet44':
    model = ResNet(n=7, shortcuts=True).to(device)
    print(f"Using ResNet-44")
elif args.model=='resnet56':
    model = ResNet(n=9, shortcuts=True).to(device)
    print(f"Using ResNet-56")
elif args.model=="vit_timm":
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(device)
    print(f"Using ViT (timm pretrained on ImageNet)")
else:
    raise ValueError(f"'{args.model}' is not a valid model")

# # For Multi-GPU
# if 'cuda' in device:
#     print(device)
#     if args.dp:
#         print("using data parallel")
#         net = torch.nn.DataParallel(net) # make parallel
#         cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint_path = './checkpoint/{}-{}-{}-ckpt.t7'.format(args.model, args.dataset, args.patch)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

# Optimizer
if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"]:
    optimizer = optim.SGD(
        model.parameters(),
        lr = lr,
        momentum = momentum,
        weight_decay = weight_decay,
        nesterov = True
    )
elif args.model == "vit_timm":
    optimizer = optim.Adam(
        model.parameters(),
        lr = lr,
        weight_decay = weight_decay
    )
else:
    raise ValueError(f"'{args.model}' is not a valid model")

 
# Scheduler
if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"]:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
elif args.model == "vit_timm":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
else:
    raise ValueError(f"'{args.model}' is not a valid model")

# Training/Evaluating Loop
print(f"Starting training — {epochs} epochs\n")

scaler = torch.cuda.amp.GradScaler()

for epoch in range(0, epochs):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, scaler
    )
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Record
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    history['lr'].append(current_lr)

    print(
        f"Epoch [{epoch:3d}/{epochs}] | LR: {current_lr:.4f} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )

    # Save record to csv file
    record_df = pd.DataFrame(history)
    record_df.to_csv(os.path.join(results_path, f'training_history_{args.model}.csv'), index=False)

    # Save best checkpoint
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        save_checkpoint(
            state={
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            },
            save_path=checkpoint_path
        )

# Plot training curves
print(f"\nTraining complete.")
print(f"Best test accuracy : {best_test_acc:.2f}%")
print(f"Best checkpoint    : {checkpoint_path}\n")

plot_history(
    history,
    milestones=milestone if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"] else None,
    save_path = os.path.join(results_path, f'training_curves_{args.model}.png') 
)