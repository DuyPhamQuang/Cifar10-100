import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#from utils import accuracy


len_testset = 10000
total_training_samples = 50000

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

def sigma_regularization(model, mode='neg_log_sum'):
        total = 0.0
        for name, module in model.named_modules():
            rho = getattr(module, "rho_kernel", None)
            if rho is None:
                rho = getattr(module, "rho_weight", None)
            if rho is None:
                continue

            sigma = torch.log1p(torch.exp(rho))  # softplus(rho), NOT detached

            if mode == "sum":
                total += sigma.sum()
            elif mode == "neg_log_sum":
                total -= torch.log(sigma + 1e-12).sum()

        return total

def train_one_epoch_bayesian(model, train_loader, num_mc, criterion, optimizer, epoch, device, scaler, tb_writer):

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # switch to train mode
    model.train()
    pbar = tqdm(train_loader, desc="Train", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        # compute output
        output_ = []
        kl_ = []
        for mc_run in range(num_mc):
            output, kl = model(images)
            output_.append(output)
            kl_.append(kl)
            #print(f"Train MC Run {mc_run+1}/{num_mc}: Output shape: {output.shape}, Output predictions: {output.argmax(dim=1)}")

        batch_size     = images.size(0)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        cross_entropy_loss = criterion(output, labels)
        scaled_kl = kl / batch_size  # Scale KL divergence by batch size 
        sigma_reg = sigma_regularization(model, mode='neg_log_sum') / batch_size  # Scale sigma regularization by batch size
        #ELBO loss
        loss = cross_entropy_loss + scaled_kl + sigma_reg

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        total_loss    += loss.item() * batch_size
        _, preds       = output.max(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = 100.0 * total_correct / total_samples

        # Update progress bar with running stats
        pbar.set_postfix({
            'loss': f'{train_loss:.4f}',
            'acc':  f'{train_acc:.2f}%',
            'sigma_sum': f'{sigma_reg.item():.4f}',
            'kl_div': f'{scaled_kl.item():.4f}',
            'cross_entropy_loss': f'{cross_entropy_loss.item():.4f}'
        })

        if tb_writer is not None:
            tb_writer.add_scalar('train/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', train_loss, epoch)
            tb_writer.add_scalar('train/accuracy', train_acc, epoch)
            tb_writer.add_scalar('train/sigma_sum', sigma_reg.item(), epoch)
            tb_writer.flush()

    return train_acc, train_loss

@torch.no_grad()
def validate_bayesian(model, val_loader, num_mc, criterion, epoch, device, tb_writer):

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    
    # switch to evaluate mode
    model.eval()
    pbar = tqdm(val_loader, desc="Validate", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # compute output
        output_ = []
        kl_ = []
        for mc_run in range(num_mc):
            output, kl = model(images)
            output_.append(output)
            kl_.append(kl)
            #print(f"Validation MC Run {mc_run+1}/{num_mc}: Output shape: {output.shape}, Output predictions: {output.argmax(dim=1)}")

        batch_size = images.size(0) 
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        cross_entropy_loss = criterion(output, labels)
        scaled_kl = kl / batch_size
        sigma_reg = sigma_regularization(model, mode='neg_log_sum') / batch_size
        #ELBO loss
        loss = cross_entropy_loss + scaled_kl + sigma_reg

        # measure accuracy and record loss
        total_loss    += loss.item() * batch_size
        _, preds       = output.max(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        test_acc = 100.0 * total_correct / total_samples
        test_loss = total_loss / total_samples

        # Update progress bar with running stats
        pbar.set_postfix({
            'loss': f'{test_loss:.4f}',
            'acc':  f'{test_acc:.2f}%',
            'sigma_sum': f'{sigma_reg.item():.4f}',
            'kl_div': f'{scaled_kl.item():.4f}',
            'cross_entropy_loss': f'{cross_entropy_loss.item():.4f}'
        })

        
        if tb_writer is not None:
            tb_writer.add_scalar('val/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('val/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('val/elbo_loss', test_loss, epoch)
            tb_writer.add_scalar('val/accuracy', test_acc, epoch)
            tb_writer.add_scalar('val/sigma_sum', sigma_reg.item(), epoch)
            tb_writer.flush()

    return test_acc, test_loss

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

@torch.no_grad()
def evaluate_bayesian(model, test_loader, num_monte_carlo):
    model.eval()

    outputs_all = []  # will hold [num_mc, batch_size, num_classes] chunks
    labels_all = []   # will hold [batch_size] chunks

    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        else:
            data, target = data.cpu(), target.cpu()

        output_mc = []
        for mc_run in range(num_monte_carlo):
            output, _ = model(data)
            output_mc.append(output)                    # [batch_size, num_classes]

        # stack MC runs: [num_mc, batch_size, num_classes]
        output_mc = torch.stack(output_mc, dim=0)
        outputs_all.append(output_mc.cpu())
        labels_all.append(target.cpu())

    # concatenate along batch dimension:
    # outputs_all: list of [num_mc, batch_i, num_classes] -> [num_mc, N_test, num_classes]
    outputs_all = torch.cat(outputs_all, dim=1)
    labels_all = torch.cat(labels_all, dim=0)          # [N_test]

    # softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs_all, dim=2)  # [num_mc, N_test, num_classes]

    # mean predictive distribution over MC samples
    pred_mean = probs.mean(dim=0)                     # [N_test, num_classes]
    Y_pred = torch.argmax(pred_mean, dim=1)           # [N_test]

    acc = (Y_pred.numpy() == labels_all.numpy()).mean() * 100
    print('Test accuracy:', acc)

    return probs, labels_all
