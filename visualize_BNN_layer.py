"""
Load the best saved Bayesian model checkpoint and summarize mu/sigma of
every layer in a single table

Produces:
  - layer_summary_<epoch>.csv     (raw numbers, for further analysis)
  - layer_summary_<epoch>.html    (color-coded table, open in a browser)
"""

import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd

from models.bayesian.resnet_variational import ResNet


# ----------------------------------------------------------------------
# 1. Extract mu / sigma from a Bayesian module
# ----------------------------------------------------------------------

def get_mu_sigma(module):
    mu = getattr(module, "mu_kernel", None)
    rho = getattr(module, "rho_kernel", None)

    if mu is None or rho is None:
        mu = getattr(module, "mu_weight", None)
        rho = getattr(module, "rho_weight", None)

    if mu is None or rho is None:
        return None, None

    mu = mu.detach().cpu()
    sigma = F.softplus(rho.detach().cpu())
    return mu, sigma


# ----------------------------------------------------------------------
# 2. Build the per-layer summary table
# ----------------------------------------------------------------------

def build_summary_table(model):
    rows = []
    for name, module in model.named_modules():
        mu, sigma = get_mu_sigma(module)
        if mu is None:
            continue

        rows.append({
            "layer": name,
            "n_weights": mu.numel(),
            "mu_mean": mu.mean().item(),
            "mu_std": mu.std().item(),
            "mu_min": mu.min().item(),
            "mu_max": mu.max().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_std": sigma.std().item(),
            "sigma_min": sigma.min().item(),
            "sigma_max": sigma.max().item(),
        })

    df = pd.DataFrame(rows).sort_values("sigma_mean", ascending=False).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 3. Save as CSV + a color-coded HTML table + console pretty-print
# ----------------------------------------------------------------------

def save_summary_table(df, out_dir, epoch=None):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_epoch{epoch}" if epoch is not None else ""

    csv_path = os.path.join(out_dir, f"layer_summary{tag}.csv")
    df.to_csv(csv_path, index=False)

    styled = (
        df.style
        .background_gradient(subset=["sigma_mean", "sigma_max", "sigma_std"], cmap="Reds")
        .background_gradient(subset=["mu_mean", "mu_std"], cmap="Blues")
        .format(precision=4)
        .set_caption(f"Bayesian layer mu/sigma summary{(' — epoch ' + str(epoch)) if epoch is not None else ''}")
    )
    html_path = os.path.join(out_dir, f"layer_summary{tag}.html")
    styled.to_html(html_path)

    print(f"Saved CSV   -> {csv_path}")
    print(f"Saved table -> {html_path}  (open in a browser for the color-coded view)\n")

    try:
        print(df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(df.to_string(index=False))

    return csv_path, html_path


# ----------------------------------------------------------------------
# 4. CLI entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR10/100 BNN weight summary table")
    parser.add_argument('--model', default='resnet32_bayesian')
    parser.add_argument('--datadir', default='data/cifar10', type=str, help='dataset to use (cifar10 or cifar100)')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10 or cifar100)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                         help='use tensorboard for logging and visualization of training progress')

    args = parser.parse_args()

    print(f"Arguments:\n{args}\n")
    print("Start running ...\n")

    results_path = f'results/Resnet/{args.dataset}' if args.model in ["resnet20", "resnet32", "resnet44", "resnet56"] \
        else f'results/{args.model}/{args.dataset}'
    os.makedirs(results_path, exist_ok=True)

    checkpoint_path = os.path.join(results_path, f'best_model_{args.model}_{args.dataset}.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    print('==> Building model..')
    if args.model == 'resnet20_bayesian':
        model = ResNet(n=3, shortcuts=True).to(device)
        print("Using ResNet-20_bayesian")
    elif args.model == 'resnet32_bayesian':
        model = ResNet(n=5, shortcuts=True).to(device)
        print("Using ResNet-32_bayesian")
    elif args.model == 'resnet44_bayesian':
        model = ResNet(n=7, shortcuts=True).to(device)
        print("Using ResNet-44_bayesian")
    elif args.model == 'resnet56_bayesian':
        model = ResNet(n=9, shortcuts=True).to(device)
        print("Using ResNet-56_bayesian")
    else:
        raise ValueError(f"'{args.model}' is not a valid model")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    out_dir = os.path.join(results_path, "weight_summary")
    df = build_summary_table(model)
    save_summary_table(df, out_dir=out_dir, epoch=checkpoint.get('epoch'))