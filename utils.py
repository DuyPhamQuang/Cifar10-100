import os
import torch
import matplotlib
matplotlib.use('Agg')     
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def save_checkpoint(state, save_path):
    """
    Saves a training checkpoint to disk.

    state dict should contain:
        {
            'epoch':                int,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc':             float,
        }

    Parameters
    ----------
    state     : dict         — checkpoint contents
    save_path : str or Path  — file path for the saved .pth checkpoint
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(state, save_path)
    print(f" Checkpoint saved → {save_path}  (epoch {state['epoch']}, "
          f"test acc {state['test_acc']:.2f}%)")


def load_checkpoint(model, optimizer, load_path, device):
    """
    Loads a checkpoint from disk and restores model + optimizer state.

    Parameters
    ----------
    model     : nn.Module    — model to load weights into
    optimizer : Optimizer    — optimizer to restore state into
                               (pass None if loading for inference only)
    load_path : str or Path  — path to the .pth checkpoint file
    device    : torch.device

    Returns
    -------
    start_epoch : int    — the epoch to resume from (saved epoch + 1)
    best_acc    : float  — best test accuracy recorded in the checkpoint
    """
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"No checkpoint found at '{load_path}'")

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_acc    = checkpoint.get('test_acc', 0.0)

    print(f"  ✓ Checkpoint loaded ← {load_path}  "
          f"(resuming from epoch {start_epoch}, best acc {best_acc:.2f}%)")
    return start_epoch, best_acc



def plot_history(history, milestones=None, save_path='./results/training_curves.png', model_name=None, dataset_name=None):
    """
    Plots accuracy, loss, and learning rate curves from a training history dict.
    Saves the figure to disk.

    Parameters
    ----------
    history    : dict  — keys: 'train_loss', 'train_acc',
                                'test_loss',  'test_acc', 'lr'
                         values: lists of per-epoch scalars
    milestones : list  — epoch numbers where LR was dropped (drawn as red
                         dashed vertical lines on accuracy and loss plots)
    save_path  : str   — file path for the saved PNG figure
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name} — {dataset_name}', fontsize=14, y=1.01)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history['train_acc'], label='Train')
    axes[0].plot(epochs, history['test_acc'],  label='Test')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[1].plot(epochs, history['train_loss'], label='Train')
    axes[1].plot(epochs, history['test_loss'],  label='Test')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cross-Entropy Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Learning Rate ─────────────────────────────────────────────────────────
    axes[2].plot(epochs, history['lr'], color='green', label='LR')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')       # log scale makes step drops clearly visible
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Draw LR-drop markers on accuracy and loss plots
    if milestones:
        for ax in axes[:2]:
            for i, m in enumerate(milestones):
                ax.axvline(
                    x=m,
                    color='red',
                    linestyle='--',
                    linewidth=0.9,
                    label='LR drop' if i == 0 else ''
                )
            ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves saved → {save_path}")


def plot_accuracy(models, y_col, title, y_label, filename, COLORS, MILESTONES):
    fig, ax = plt.subplots(figsize=(10, 5))

    for (name, df), color in zip(models.items(), COLORS):
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df[y_col], label=name, color=color, linewidth=1.8)

    for i, m in enumerate(MILESTONES):
        ax.axvline(x=m, color='gray', linestyle='--', linewidth=0.9,
                   label='LR drop' if i == 0 else None)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


# def accuracy(output, target, topk=(1, )):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
