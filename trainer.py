import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#from utils import accuracy


len_testset = 10000

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

        batch_size     = images.size(0)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        cross_entropy_loss = criterion(output, labels)
        scaled_kl = kl / batch_size 
        #ELBO loss
        loss = cross_entropy_loss + scaled_kl

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        total_loss    += loss.item() * batch_size
        _, preds       = output.max(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        # Update progress bar with running stats
        pbar.set_postfix({
            'loss': f'{total_loss / total_samples:.4f}',
            'acc':  f'{100.0 * total_correct / total_samples:.2f}%'
        })

        if tb_writer is not None:
            tb_writer.add_scalar('train/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', loss.item(), epoch)
            tb_writer.add_scalar('train/accuracy', 100.0 * total_correct / total_samples, epoch)
            tb_writer.flush()

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

        batch_size = images.size(0) 
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        cross_entropy_loss = criterion(output, labels)
        scaled_kl = kl / batch_size 
        #ELBO loss
        loss = cross_entropy_loss + scaled_kl

        # measure accuracy and record loss
        total_loss    += loss.item() * batch_size
        _, preds       = output.max(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        # Update progress bar with running stats
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc':  f'{100.0 * total_correct / total_samples:.2f}%'
        })

        test_acc = 100.0 * total_correct / total_samples
        test_loss = total_loss / total_samples

        if tb_writer is not None:
            tb_writer.add_scalar('val/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('val/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('val/elbo_loss', loss.item(), epoch)
            tb_writer.add_scalar('val/accuracy', 100.0 * total_correct / total_samples, epoch)
            tb_writer.flush()

    return test_acc

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

def evaluate_bayesian( model, test_loader, num_monte_carlo):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    output_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_mc = []
            for mc_run in range(num_monte_carlo):
                output, _ = model.forward(data)
                output_mc.append(output)
            output_ = torch.stack(output_mc)
            output_list.append(output_)
            labels_list.append(target)

        output = torch.stack(output_list)
        output = output.permute(1, 0, 2, 3)
        output = output.contiguous().view(num_monte_carlo, len_testset, -1)
        output = torch.nn.functional.softmax(output, dim=2)
        labels = torch.cat(labels_list)
        pred_mean = output.mean(dim=0)
        Y_pred = torch.argmax(pred_mean, axis=1)
        print('Test accuracy:',
              (Y_pred.data.cpu().numpy() == labels.data.cpu().numpy()).mean() *
              100)

        return output, labels
