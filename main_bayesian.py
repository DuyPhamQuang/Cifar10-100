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
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from models.bayesian.resnet_variational import ResNet
from models.vit import VisionTransformer
from data_loader import get_data_loaders, plot_images
from trainer import train_one_epoch, evaluate, train_one_epoch_bayesian, validate_bayesian, evaluate_bayesian
from utils import save_checkpoint, load_checkpoint, plot_history, plot_accuracy

from bayesian_torch.utils.util import get_rho

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_pareto_front

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

parser.add_argument('--mode', type=str, required=True, help='train | test | HPs_tuning | reconstruct | measure_uncertainty')
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
#gamma = args.gamma

# Paths
results_path = f'results/Resnet/{args.dataset}' if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"] \
                                else f'results/{args.model}/{args.dataset}'
os.makedirs(results_path, exist_ok=True)

checkpoint_path = os.path.join(results_path, f'best_model_{args.model}_{args.dataset}.pth')

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
else:
    raise ValueError(f"'{args.model}' is not a valid model")

logger_dir = os.path.join(args.log_dir, f"{args.model}_{args.dataset}")
tb_writer = SummaryWriter(logger_dir)

@torch.no_grad()
def get_mean_sigma(model):
    sigmas = []
    for _, module in model.named_modules():
        rho = getattr(module, "rho_kernel", None)
        if rho is None:
            rho = getattr(module, "rho_weight", None)
        if rho is None:
            continue
        sigmas.append(torch.log1p(torch.exp(rho.detach()) + 1e-12).flatten())
    return torch.cat(sigmas).mean().item() if sigmas else 0.0

@torch.no_grad()
def log_layer_uncertainty(model, writer, epoch):
    for name, module in model.named_modules():
        mu = getattr(module, "mu_kernel", None)
        rho = getattr(module, "rho_kernel", None)

        if mu is None or rho is None:
            mu = getattr(module, "mu_weight", None)
            rho = getattr(module, "rho_weight", None)

        if mu is None or rho is None:
            continue

        mu = mu.detach()
        sigma = torch.log1p(torch.exp(rho.detach()))
        snr = mu.abs() / (sigma + 1e-12)

        writer.add_histogram(f"{name}/mu", mu, epoch)
        writer.add_histogram(f"{name}/sigma", sigma, epoch)
        writer.add_histogram(f"{name}/snr", snr, epoch)
        writer.add_scalar(f"{name}/mean_sigma", sigma.mean().item(), epoch)
        writer.add_scalar(f"{name}/mean_snr", snr.mean().item(), epoch)

 
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

        # per-layer weight uncertainty logging
        log_layer_uncertainty(model, tb_writer, epoch)


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

if args.mode == "HPs_tuning":
    from trainer import train_one_epoch_bayesian_with_HPs, validate_bayesian_for_tuning_process

    VAL_SIZE = 5000
    NUM_MC_EVAL = 10  
    HPO_EPOCHS = 150           # reduced-epoch proxy budget per Optuna trial
    FULL_EPOCHS = 300         # final full-length training run
    N_TRIALS = 20
    seed = 42

    # --------------------------------------------------------------------------
    # Data: train / val split
    # --------------------------------------------------------------------------
    train_dataset = datasets.CIFAR10(root=datadir, train=True, download=True, transform=train_transform) if args.dataset == 'cifar10' \
        else datasets.CIFAR100(root=datadir, train=True, download=True, transform=train_transform)

    validation_dataset = datasets.CIFAR10(root=datadir, train=True, download=True, transform=test_transform) if args.dataset == 'cifar10' \
        else datasets.CIFAR100(root=datadir, train=True, download=True, transform=test_transform)

    n_total = len(validation_dataset)
    n_train = n_total - VAL_SIZE
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, VAL_SIZE], generator=generator)

    train_subset = torch.utils.data.Subset(train_dataset, train_idx.indices)
    val_subset = torch.utils.data.Subset(validation_dataset, val_idx.indices)

    train_loader_2 = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --------------------------------------------------------------------------
    # Optuna multi-objective search: maximize (val_acc, mean_sigma)
    # --------------------------------------------------------------------------
    def objective(trial, model, optimizer, train_loader, val_loader):
        beta = trial.suggest_float("beta", 1e-3, 1.0, log=True)
        lambda_sigma = trial.suggest_float("lambda_sigma", 1e-5, 1e-1, log=True)

        for epoch in range(HPO_EPOCHS):
            train_one_epoch_bayesian_with_HPs(model, train_loader, num_mc=1, criterion=criterion, optimizer=optimizer, beta=beta, lambda_sigma=lambda_sigma, epoch=epoch, device=device)
            val_acc = validate_bayesian_for_tuning_process(model, val_loader, num_mc=3, criterion=criterion, epoch=epoch, device=device)  # cheap MC count while pruning
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_val_acc = validate_bayesian_for_tuning_process(model, val_loader, num_mc=NUM_MC_EVAL, criterion=criterion, epoch=HPO_EPOCHS, device=device)
        mean_sigma = get_mean_sigma(model)
        #trial.set_user_attr("mean_sigma_raw", mean_sigma)
        return final_val_acc, mean_sigma


    def run_hpo(train_loader, val_loader):
        sampler = TPESampler(n_startup_trials=5)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        STORAGE_PATH = f"sqlite:///optuna_studies/bayesian_hpo_{args.model}_{args.dataset}.db"
        study = optuna.create_study(
            directions=["maximize", "maximize"], sampler=sampler, pruner=pruner,
            storage=STORAGE_PATH,
            study_name=f"bayesian_hpo_{args.model}_{args.dataset}",
            load_if_exists=True
        )
        study.optimize(lambda t: objective(t, model, optimizer, train_loader, val_loader), n_trials=N_TRIALS)
        return study


    # --------------------------------------------------------------------------
    # Pareto front plot + pick a configuration
    # --------------------------------------------------------------------------
    def save_pareto_plot(study, out_html="pareto_front.html", out_png="pareto_front.png"):
        fig = plot_pareto_front(study, target_names=["Val Accuracy", "Mean Sigma"])
        fig.write_html(out_html)

        fig.write_image(out_png)  # requires `kaleido` installed


    def pick_config(study, min_accuracy=None):
        candidates = study.best_trials
        if min_accuracy is not None:
            filtered = [t for t in candidates if t.values[0] >= min_accuracy]
            candidates = filtered if filtered else candidates
        best = max(candidates, key=lambda t: t.values[1])  # highest mean_sigma among candidates
        return best.params


    # --------------------------------------------------------------------------
    # Final full-length training run with chosen hyperparameters on the pre-splitting
    # training dataset
    # --------------------------------------------------------------------------
    def final_training(params, model, optimizer, train_loader, test_loader,
                    epochs=FULL_EPOCHS, tb_writer=None):
        beta = params["beta"]
        lambda_sigma = params["lambda_sigma"]

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch_bayesian_with_HPs(model, train_loader, num_mc=1, criterion=criterion, optimizer=optimizer, beta=beta, lambda_sigma=lambda_sigma, epoch=epoch, device=device)
            test_acc = validate_bayesian_for_tuning_process(model, test_loader, num_mc=NUM_MC_EVAL, criterion=criterion, epoch=epoch, device=device)
            mean_sigma = get_mean_sigma(model)

            tb_writer.add_scalar("train_with_HPs/loss", train_loss, epoch)
            tb_writer.add_scalar("train_with_HPs/accuracy", train_acc, epoch)
            tb_writer.add_scalar("val_with_HPs/accuracy", test_acc, epoch)
            tb_writer.add_scalar("val_with_HPs/mean_sigma", mean_sigma, epoch)

            if epoch % 5 == 0 or epoch == epochs - 1:
                log_layer_uncertainty(model, tb_writer, epoch)

            print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f}% val_acc={test_acc:.2f}% mean_sigma={mean_sigma:.4f}")

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

        #test_acc = validate_bayesian_for_tuning_process(model, test_loader, num_mc=NUM_MC_EVAL, criterion=criterion, epoch=epochs, device=device)
        #print(f"\nFinal held-out TEST accuracy: {test_acc:.2f}%")
        #tb_writer.add_scalar("test_with_HPs/accuracy", test_acc, epochs)
        tb_writer.flush()
        tb_writer.close()
        #return model, best_test_acc


    # --------------------------------------------------------------------------
    # Orchestration
    # --------------------------------------------------------------------------
    print("Running multi-objective Optuna search (val_acc, mean_sigma)...")
    study = run_hpo(train_loader_2, val_loader)

    print(f"\nPareto-optimal trials ({len(study.best_trials)}):")
    for t in study.best_trials:
        print(f"  beta={t.params['beta']:.5f}  lambda_sigma={t.params['lambda_sigma']:.6f} "
            f"-> val_acc={t.values[0]:.2f}%  mean_sigma={t.values[1]:.4f}")

    out_html = os.path.join(results_path, 'pareto_front.html')
    out_png = os.path.join(results_path, 'pareto_front.png')
    save_pareto_plot(study, out_html, out_png)

    chosen_params = pick_config(study, min_accuracy=79)  # adjust accuracy floor as needed
    print(f"\nChosen configuration: {chosen_params}")

    final_training(chosen_params, model, optimizer, train_loader, test_loader, epochs=FULL_EPOCHS, tb_writer=tb_writer)

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

    np.save(f'{results_path}/probs_cifar_mc_{args.dataset}.npy', output.data.cpu().numpy())
    np.save(f'{results_path}/cifar_test_labels_mc_{args.dataset}.npy', labels.data.cpu().numpy())

if args.mode == "measure_uncertainty":
    # Load the best model checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze the model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ### Measure on the training set
    print(f"Measuring uncertainty on the training set ...")

    batch = 0
    batch_limit = 20
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Multiple MC forward passes
        probs_mc = []
        for _ in range(args.num_monte_carlo):
            outputs, _ = model(images)
            probs_mc.append(F.softmax(outputs, dim=1))
        probs_mc = torch.stack(probs_mc, dim=0)          # [num_mc, N, C]

        # Per-candidate uncertainty: variance across MC samples, summed over classes
        var_probs = probs_mc.var(dim=0).sum(dim=1)       # [N]
        uncertainty_loss = var_probs.mean()

        batch += 1
        print(f"batch {batch}  | mean_uncertainty={uncertainty_loss.item():.6f}")

        # logging with tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar('uncertainty/mean_uncertainty_training_set', uncertainty_loss.item(), batch)

        if batch >= batch_limit:
            break

    ### Measure uncertainty on random inputs
    print(f"Measuring uncertainty on random inputs ...")

    num_candidates = 1

    # Initialize random data points for reconstruction
    x0, y0 = next(iter(train_loader))
    print('X:', x0.shape, x0.device)
    print('y:', y0.shape, y0.device)

    n, c, h, w = x0.shape
    x_raw = torch.rand(num_candidates, c, h, w).to(device)

    # Apply transform
    mean_t = torch.tensor(mean).view(1, c, 1, 1)
    std_t = torch.tensor(std).view(1, c, 1, 1)

    x_raw_transform = (x_raw - mean_t) / std_t

    # Multiple MC forward passes
    probs_mc = []
    for _ in range(args.num_monte_carlo):
        outputs, _ = model(x_raw_transform)
        probs_mc.append(F.softmax(outputs, dim=1))
    probs_mc = torch.stack(probs_mc, dim=0)          # [num_mc, N, C]

    # Per-candidate uncertainty: variance across MC samples, summed over classes
    var_probs = probs_mc.var(dim=0).sum(dim=1)       # [N]
    uncertainty_loss = var_probs.mean()

    print(f"random batch of images  | mean_uncertainty={uncertainty_loss.item():.6f}")

    # logging with tensorboard
    if tb_writer is not None:
        tb_writer.add_scalar('uncertainty/mean_uncertainty_random_images', uncertainty_loss.item(), batch)


if args.mode == "reconstruct":
    print(f"Reconstructing training set from {args.dataset} based on achieved weights ...")

    # Hyperparameters
    num_candidates = 1
    reconstruction_epochs = 10000
    # real label
    #_, real_label = next(iter(train_loader))


    # Support utils function
    @torch.no_grad()
    def calc_model_parameters(model):
        l = [torch.tensor(p.shape).prod() for p in model.parameters()]
        print('Parameters per Layer:', l)
        print('Total Parameters:', torch.tensor(l).sum().item())

    # Load the best model checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze the model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Initialize random data points for reconstruction
    x0, y0 = next(iter(train_loader))
    print('X:', x0.shape, x0.device)
    print('y:', y0.shape, y0.device)

    n, c, h, w = x0.shape
    x_raw = torch.rand(num_candidates, c, h, w).to(device)
    x_raw.requires_grad_(True)
    opt_x = torch.optim.SGD([x_raw], lr=1e-4, momentum=0.9)

    # For the reconstruction, we will use the same normalization as used during training
    mean_t = torch.tensor(mean).view(1, c, 1, 1)
    std_t = torch.tensor(std).view(1, c, 1, 1)

    pbar = tqdm(range(reconstruction_epochs), desc="Reconstruction Progress", unit="step")

    for step in pbar:
        opt_x.zero_grad()

        # Apply transform
        x_input = (x_raw - mean_t) / std_t

        # Multiple MC forward passes
        probs_mc = []
        for _ in range(args.num_monte_carlo):
            outputs, _ = model(x_input)
            probs_mc.append(F.softmax(outputs, dim=1))
        probs_mc = torch.stack(probs_mc, dim=0)          # [num_mc, N, C]

        # Per-candidate uncertainty: variance across MC samples, summed over classes
        var_probs = probs_mc.var(dim=0).sum(dim=1)       # [N]
        uncertainty_loss = var_probs.mean()

        loss = uncertainty_loss
        loss.backward()
        opt_x.step()

        # Keep pixel values in valid image range
        with torch.no_grad():
            x_raw.clamp_(0.0, 1.0)

        pbar.set_postfix({
            'mean_uncertainty': f'{uncertainty_loss.item():.6f}',
            'min_uncertainty': f'{var_probs.min().item():.6f}',
            'max_uncertainty': f'{var_probs.max().item():.6f}',
        })

        # logging with tensor
        if tb_writer is not None:
            tb_writer.add_scalar('reconstruction/mean_uncertainty', uncertainty_loss.item(), step)

        if step % 200 == 0 or step == reconstruction_epochs - 1:
            # history["step"].append(step)
            # history["mean_uncertainty"].append(uncertainty_loss.item())
            # history["per_candidate_uncertainty"].append(var_probs.detach().cpu().clone())

            # print(f"step {step:5d} | mean_uncertainty={uncertainty_loss.item():.6f} "
            #       f"| min={var_probs.min().item():.6f} max={var_probs.max().item():.6f}")
            
            
            grid = make_grid(x_raw.detach().cpu(), nrow=5, padding=2)
            save_image(grid, f'{results_path}/candidates_step{step:05d}.png')    