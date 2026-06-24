# ======================================================================
# ResNet implementation for Cifar10 as specified in Section 4.2 
# of "Deep Residual Learning for Image Recognition" (He et al., 2015)
# with Bayesian variational layers on top of the standard ResNet blocks.
# ======================================================================

import torch
from torch import nn
import torch.nn.functional as F
from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization


# per-weight prior and posterior initialisation parameters for variational layers
prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class Block(nn.Module):
    """
    A 2-layer residual learning building block as illustrated by Fig.2 
    in "Deep Residual Learning for Image Recognition" (He et al., 2015)".

    Architecture: Conv -> BN -> ReLU -> Conv -> BN -> (+shortcut) -> ReLU

    Parameters:
    - in_channels:  int,  number of input channels
    - out_channels: int,  number of output channels
    - subsample:    bool, if True, stride=2 on conv1 to halve spatial dims
                         and double channels
    """
    def __init__(self, in_channels, out_channels, subsample=False):
        super().__init__()

        stride = 2 if subsample else 1

        # Main path
        self.conv1 = Conv2dReparameterization(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=stride, padding=1, 
            bias=False,
            prior_mean=prior_mu, 
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init)
        
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = Conv2dReparameterization(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=1, 
            padding=1, 
            bias=False,
            prior_mean=prior_mu, 
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init)
        
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.subsample = subsample

        # He initialisation (Kaiming normal) for conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _project_shortcut(self, x):
        """
        Identity shortcut projection when dimensions differ
        When spatial dims and channels differ:
          1. Subsample x spatially with stride 2
          2. Zero-pad channels -> double channels with zeros
        """
        # Subsample spatially: stride=2 by taking every other pixel
        x_down = x[:, :, ::2, ::2]
        # Zero-pad channel dimension to match out_channels
        # x_down has C channels, need to pad C more channels with zeros to get 2C channels
        pad = torch.zeros_like(x_down)
        return torch.cat((x_down, pad), dim=1)

    def forward(self, x, use_shortcut=True):
        kl_sum = 0
        z, kl = self.conv1(x)
        kl_sum += kl
        z = self.bn1(z)
        z = self.relu1(z)

        z, kl = self.conv2(z)
        kl_sum += kl
        z = self.bn2(z)

        if use_shortcut:
            if self.subsample or (x.shape != z.shape):
                shortcut = self._project_shortcut(x)
            else:
                shortcut = x         
            z = z + shortcut

        z = self.relu2(z)
        return z, kl_sum


class ResNet(nn.Module):
    """
    Intended ResNet architecture for Cifar10 as specified in Section 4.2 of
    "Deep Residual Learning for Image Recognition" (He et al., 2015).

    Architecture:
      - 1 entry conv (3x3, 16channels, stride 1)
      - 3 stacks of n blocks each:
          Stack 1: 32x32, 16 channels
          Stack 2: 16x16, 32 channels  (first block subsamples)
          Stack 3:  8x8,  64 channels  (first block subsamples)
      - Global average pool → FC → output

    Depth formula: 6n + 2 total weight layers
      n=3  → ResNet-20  
      n=5  → ResNet-32  
      n=7  → ResNet-44 
      n=9  → ResNet-56  
      n=18 → ResNet-110

    Parameters:
    - n:           int,  number of blocks per stack 
    - num_classes: int,  output classes (default 10 for Cifar10)
    - shortcuts:   bool, if False → plain network
    """
    def __init__(self, n, num_classes=10, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts

        # Entry layer
        self.conv_in = Conv2dReparameterization(
            in_channels=3, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False,
            prior_mean=prior_mu, 
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init)
        self.bn_in   = nn.BatchNorm2d(16)
        self.relu_in = nn.ReLU(inplace=True)

        # Stack 1: 32×32, 16 channels, n blocks, all stride 1
        self.stack1 = nn.ModuleList([
            Block(16, 16, subsample=False) for _ in range(n)
        ])

        # Stack 2: 16×16, 32 channels, n blocks
        # First block subsamples (stride 2) and doubles channels 16→32
        self.stack2 = nn.ModuleList(
            [Block(16, 32, subsample=True)] +
            [Block(32, 32, subsample=False) for _ in range(n - 1)]
        )

        # Stack 3: 8×8, 64 channels, n blocks
        # First block subsamples (stride 2) and doubles channels 32→64
        self.stack3 = nn.ModuleList(
            [Block(32, 64, subsample=True)] +
            [Block(64, 64, subsample=False) for _ in range(n - 1)]
        )

        # Exit
        # global average pooling, a 10-way fc layer, and softmax
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearReparameterization(
            in_features=64, 
            out_features=num_classes,
            prior_mean=prior_mu, 
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )

        # He initialisation for the FC layer
        self.apply(_weights_init)
        

    def forward(self, x):
        kl_sum = 0
        # Entry
        z, kl = self.conv_in(x)
        kl_sum += kl
        z = self.bn_in(z)
        z = self.relu_in(z)

        # Stacks
        for block in self.stack1:
            z, kl = block(z, use_shortcut=self.shortcuts)
            kl_sum += kl
        for block in self.stack2:
            z, kl = block(z, use_shortcut=self.shortcuts)
            kl_sum += kl
        for block in self.stack3:
            z, kl = block(z, use_shortcut=self.shortcuts)
            kl_sum += kl

        # Exit
        z = self.avgpool(z)
        z = z.flatten(1)
        z, kl = self.fc(z)
        kl_sum += kl
        return z, kl_sum