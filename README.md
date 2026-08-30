# CIFAR Benchmark Study

> **Note:** This repository is a work in progress. Results and reproductions are being added incrementally. The primary development work is available on the [`mainv2`](../../tree/mainv2) branch.

## Overview

This repository benchmarks state-of-the-art (SOTA) image-classification models on the **CIFAR-10** and **CIFAR-100** datasets. It covers a range of architectures, from established convolutional neural networks to more recent Transformer-based approaches.

- **CNN-based models:** ResNet, Wide ResNet (WRN), and more
- **Vision Transformer-based models:** Vision Transformer (ViT), and more

## Results

A summary of the benchmark results reported for each examined paper is available in [`Report.md`](./Report.md).

## Reproductions

A subset of the reviewed models is reproduced either through from-scratch implementations or by adapting well-known public repositories. Each reproduction includes an appropriate reference to its original source.

The primary entry point for standard model training and evaluation is [`main.py`](./main.py). Model definitions, training and validation loops, data-loading logic, and utility functions can be traced through the imports in this file.

### Example commands

The available command-line arguments can be inferred from the argument-parsing section of `main.py`. The following commands illustrate typical training runs.

#### Training

```bash
uv run main.py --mode train --datadir data/cifar10 --dataset cifar10 --model resnet32 --epochs 300

uv run main.py --mode train --datadir data/cifar10 --dataset cifar10 --model vit --epochs 300

uv run main.py --mode train --datadir data/cifar10 --dataset cifar10 --model vit_timm --epochs 300
```

## Reconstruction Training Dataset Using BNN Uncertainty

This part of the repository investigates the process of reconstructing training dataset based on uncertainty estimates produced by Bayesian neural networks (BNNs).

The main entry point is [`main_bayesian.py`](./main_bayesian.py). Additional scripts include:

- [`visualize_BN_layer.py`](./visualize_BN_layer.py) for visualizing Bayesian-layer weight distributions
- [`visualize_bnn_prediction.py`](./visualize_bnn_prediction.py) for visualizing predictive uncertainty
- `models/bayesian/resnet_variational.py` for the Bayesian ResNet model implementation

The BNN training and validation loops, along with data-loading and utility functions, can be traced through the imports in `main_bayesian.py`.

### Example commands

The available command-line arguments can be inferred from the argument-parsing section of `main_bayesian.py`. The commands below demonstrate the main workflows.

#### Training

```bash
uv run main_bayesian.py --mode train --datadir data/cifar10 --dataset cifar10 --model resnet32_bayesian --epochs 300
```

#### Testing and uncertainty visualization

Run the model in test mode to visualize selected predictive-uncertainty outputs.

```bash
uv run main_bayesian.py --mode test --datadir data/cifar10 --dataset cifar10 --model resnet32_bayesian
```

#### Hyperparameter tuning

Hyperparameter tuning is implemented with [Optuna](https://optuna.org/). The current sampler is the Tree-structured Parzen Estimator (TPE) sampler.

The current tuning results are still being improved. Possible directions include:

- Expanding or refining the hyperparameter search ranges
- Evaluating alternative Optuna samplers
- Increasing the number of training epochs used during trials
- Revisiting the tuning objective and evaluation settings
Vice versa

```bash
uv run main_bayesian.py --mode HP_tunings --datadir data/cifar10 --dataset cifar10 --model resnet32_bayesian --epochs 300
```

#### Reconstruction

```bash
uv run main_bayesian.py --mode reconstruct --datadir data/cifar10 --dataset cifar10 --model resnet32_bayesian
```

#### Uncertainty measurement

```bash
uv run main_bayesian.py --mode measure_uncertainty --datadir data/cifar10 --dataset cifar10 --model resnet32_bayesian
```

#### Visualizing Bayesian-layer weight distributions

To visualize the weight distributions of each layer in the best model specified by the checkpoint path, run:

```bash
uv run visualize_BN_layer.py
```