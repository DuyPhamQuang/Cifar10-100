'''
BNN pipeline for CNN-based models on Cifar 10/100 dataset.
Take inspiration from https://github.com/IntelLabs/bayesian-torch
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
from torch.utils.tensorboard import SummaryWriter

from models.bayesian.resnet_variational import ResNet
from models.vit import VisionTransformer
from data_loader import get_data_loaders, plot_images
from trainer import train_one_epoch, evaluate, train_one_epoch_bayesian, validate_bayesian, evaluate_bayesian
from utils import save_checkpoint, load_checkpoint, plot_history, plot_accuracy

from bayesian_torch.utils.util import get_rho

# parsers
parser = argparse.ArgumentParser(description='CIFAR10/100 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--milestone', nargs='+', default=[60, 120, 160], type=int, help='milestones for MultiStepLR')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for MultiStepLR')
parser.add_argument('--optimizer', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='resnet32_bayesian')
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patch_size', default=4, type=int)
parser.add_argument('--datadir', default='data/cifar10', type=str, help='dataset to use (cifar10 or cifar100)')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10 or cifar100)')

parser.add_argument('--mode', type=str, required=True, help='train | test')
parser.add_argument('--num_monte_carlo', type=int, default=20, help='number of Monte Carlo samples to be drawn during inference')
parser.add_argument('--num_mc', type=int, default=1, help='number of Monte Carlo runs during training')
parser.add_argument('--tensorboard', type=bool, default=True, help='use tensorboard for logging and visualization ' \
'of training progress')
parser.add_argument( '--log_dir', type=str, default='./logs', help='use tensorboard for logging '
'and visualization of training progress')


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

# if args.model == "vit_timm" and args.dataset == "cifar10":   
#     print("For fine-tuning ViT on Cifar10, these default hyperparameters will be used:")
#     lr = 1e-4
#     weight_decay = 5e-4
#     epochs = 12
#     batch_size = 64
#     print(f"Learning Rate for ViT_Timm: {lr}\n")
#     print(f"Weight Decay for ViT_Timm: {weight_decay}\n")
#     print(f"Epochs for ViT_Timm: {epochs}\n")
#     print(f"Batch Size for ViT_Timm: {batch_size}\n")

# if args.model == "vit_timm" and args.dataset == "cifar100":
#     print("For fine-tuning ViT on Cifar100, these default hyperparameters will be used:")
#     lr = 1e-4
#     weight_decay = 1e-3
#     epochs = 20
#     batch_size = 64
#     print(f"Learning Rate for ViT_Timm: {lr}\n")
#     print(f"Weight Decay for ViT_Timm: {weight_decay}\n")
#     print(f"Epochs for ViT_Timm: {epochs}\n")
#     print(f"Batch Size for ViT_Timm: {batch_size}\n")

# if args.model == "vit" and args.dataset == "cifar10":
#     print("For fine-tuning ViT on Cifar10, these default hyperparameters will be used:")
#     input_channels = 3
#     patch_size = 4
#     embedding_dim = 768 
#     num_heads = 12
#     mlp_hidden_dim = 3072
#     num_blocks = 12
#     drop_out=0.1
#     lr = 3e-4
#     weight_decay = 5e-5
#     epochs = 200
#     batch_size = 128

# if args.model == "vit" and args.dataset == "cifar100":
#     print("For fine-tuning ViT on Cifar100, these default hyperparameters will be used:")
#     input_channels = 3
#     patch_size = 4
#     embedding_dim = 768 
#     num_heads = 12
#     mlp_hidden_dim = 3072
#     num_blocks = 12
#     drop_out=0.1
#     lr = 3e-4
#     weight_decay = 5e-4
#     epochs = 200
#     batch_size = 128

# Paths
results_path = f'results/Resnet/{args.dataset}' if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"] \
                                else f'results/{args.model}/{args.dataset}'
os.makedirs(results_path, exist_ok=True)

checkpoint_path = os.path.join(results_path, f'best_model_{args.model}.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# best test accuracy (for saving best model)
best_test_acc = 0.0


# # History (for plotting)
# history = {
#     'train_loss': [],
#     'train_acc': [],
#     'test_loss': [],
#     'test_acc': [],
#     'lr': []
# }

# Preparing dataset
print('==> Preparing data..')
if args.model=="vit_timm":
    size = 224
else:
    size = image_size


# Set up normalization based on the dataset
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
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

elif args.model == "vit":
    if args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:  # cifar10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

elif args.model in ["resnet20_bayesian", "resnet32_bayesian", "resnet44_bayesian", "resnet56_bayesian"]:
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
if args.model=='resnet20_bayesian':
    model = ResNet(n=3, shortcuts=True).to(device)
    print(f"Using ResNet-20_bayesian")
elif args.model=='resnet32_bayesian':
    model = ResNet(n=5, shortcuts=True).to(device)
    print(f"Using ResNet-32_bayesian")
elif args.model=='resnet44_bayesian':
    model = ResNet(n=7, shortcuts=True).to(device)
    print(f"Using ResNet-44_bayesian")
elif args.model=='resnet56_bayesian':
    model = ResNet(n=9, shortcuts=True).to(device)
    print(f"Using ResNet-56_bayesian")
# elif args.model=="vit_timm":
#     model = timm.create_model("vit_base_patch16_224", pretrained=True)
#     model.head = nn.Linear(model.head.in_features, num_classes)
#     model = model.to(device)
#     print(f"Using ViT (timm pretrained on ImageNet)")
# elif args.model=="vit":
#     model = VisionTransformer(
#         input_channels=input_channels,
#         embedding_dim=embedding_dim,
#         patch_size=patch_size,
#         image_size=image_size,
#         num_heads=num_heads,
#         mlp_hidden_dim=mlp_hidden_dim,
#         num_blocks=num_blocks,
#         num_classes=num_classes,
#         dropout=drop_out
#     ).to(device)
#     print(f"Using ViT_base (from scratch)")
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
criterion = nn.CrossEntropyLoss() # Already includes softmax, so outputs can be raw logits

# Optimizer
if args.model in ["resnet20_bayesian", "resnet32_bayesian", "resnet44_bayesian", "resnet56_bayesian"]:
    optimizer = torch.optim.Adam(model.parameters(), lr)
# elif args.model == "vit_timm" or args.model == "vit":
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr = lr,
#         weight_decay = weight_decay
#     )
else:
    raise ValueError(f"'{args.model}' is not a valid model")

logger_dir = os.path.join(args.log_dir, f"{args.model}_{args.dataset}")
tb_writer = SummaryWriter(logger_dir)

 
# Scheduler
# if args.model in ["resnet20_bayesian", "resnet32_bayesian", "resnet44_bayesian", "resnet56_bayesian"]:
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
# elif args.model == "vit_timm" or args.model == "vit":
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
# else:
#     raise ValueError(f"'{args.model}' is not a valid model")

if args.mode == "train":

    # Training/Validating Loop
    print(f"Starting training — {epochs} epochs\n")

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(0, epochs):

        train_acc, train_loss = train_one_epoch_bayesian(
            model, train_loader, args.num_mc, criterion, optimizer, epoch, device, scaler, tb_writer
        )
        test_acc, test_loss = validate_bayesian(
            model, test_loader, args.num_mc, criterion, epoch, device, tb_writer
        )

        if (epoch >= 80 and epoch < 120):
            lr = 0.1 * args.lr
        elif (epoch >= 120 and epoch < 160):
            lr = 0.01 * args.lr
        elif (epoch >= 160 and epoch < 180):
            lr = 0.001 * args.lr
        elif (epoch >= 180):
            lr = 0.0005 * args.lr

        #current_lr = optimizer.param_groups[0]['lr']

        # # Record
        # history['train_loss'].append(train_loss)
        # history['train_acc'].append(train_acc)
        # history['test_loss'].append(test_loss)
        # history['test_acc'].append(test_acc)
        # history['lr'].append(current_lr)

        print(
            f"Epoch [{epoch:3d}/{epochs}] | LR: {lr:.4f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        # # Save record to csv file
        # record_df = pd.DataFrame(history)
        # record_df.to_csv(os.path.join(results_path, f'training_history_{args.model}.csv'), index=False)

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

    print(f"\nTraining complete.")
    print(f"Best test accuracy : {best_test_acc:.2f}%")
    print(f"Best checkpoint    : {checkpoint_path}\n")

# plot_history(
#     history,
#     milestones=milestone if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"] else None,
#     save_path = os.path.join(results_path, f'training_curves_{args.model}.png'),
#     model_name=args.model,
#     dataset_name=args.dataset
# )

if args.mode == "test":
    # Evaluate the best model on the test set
    print(f"Evaluating the best model on the test set ...")
    # Load the best model checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    output, labels = evaluate_bayesian(model, test_loader, args.num_monte_carlo)

    np.save(f'{results_path}/probs_cifar_mc.npy', output.data.cpu().numpy())
    np.save(f'{results_path}/cifar_test_labels_mc.npy', labels.data.cpu().numpy())