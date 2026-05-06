import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Train", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        _, preds       = outputs.max(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        # Update progress bar with running stats
        pbar.set_postfix({
            'loss': f'{total_loss / total_samples:.4f}',
            'acc':  f'{100.0 * total_correct / total_samples:.2f}%',
        })

    avg_loss = total_loss / total_samples
    avg_acc  = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(test_loader, desc=" Val ", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        _, preds       = outputs.max(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        pbar.set_postfix({
            'loss': f'{total_loss / total_samples:.4f}',
            'acc':  f'{100.0 * total_correct / total_samples:.2f}%',
        })

    avg_loss = total_loss / total_samples
    avg_acc  = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc