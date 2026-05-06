# SOTA Image Classification on CIFAR-10 and CIFAR-100: A Summary

---

## Part 1: Pre-trained Models

These methods leverage large-scale pretraining (supervised or self-supervised) on external datasets before fine-tuning on CIFAR.

---

### 1. Vision Transformer (ViT)
**Paper**: *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*
Dosovitskiy et al. (2020) — arXiv:[2010.11929](https://arxiv.org/pdf/2010.11929)

**Method**: Applies a standard Transformer to sequences of image patches.Supervised-pretrained on Google's private **JFT-300M** dataset, making the top results inaccessible for external reproduction. The publicly accessible version uses ImageNet-21k (14M images).

| Dataset   | Model      | Pretrain Data        | Accuracy  |
|-----------|------------|----------------------|-----------|
| CIFAR-10  | ViT-H/14   | JFT-300M (private)   | **99.50%**|
| CIFAR-10  | ViT-L/16   | ImageNet-21k (public)|   99.15%  |
| CIFAR-10  | ViT-L/16   | JFT-300M (private)   | **99.42%**|
| CIFAR-100 | ViT-H/14   | JFT-300M (private)   | **94.55%**|
| CIFAR-100 | ViT-L/16   | ImageNet-21k (public)|   93.25%  |
| CIFAR-100 | ViT-L/16   | JFT-300M (private)   |   93.90%  |

>⚠️ The 99.50% (CIFAR-10) and 94.55% (CIFAR-100) results require JFT-300M, a dataset exclusive to Google and not publicly reproducible.
> Public code: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

---

### 2. Big Transfer (BiT)
**Paper**: *Big Transfer (BiT): General Visual Representation Learning* 
Kolesnikov et al. (2019) — arXiv:[1912.11370](https://arxiv.org/pdf/1912.11370) — **ECCV 2020**

**Method**: Revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. 

| Dataset   | Model             | Pretrain Data         | Accuracy |
|-----------|-------------------|-----------------------|----------|
| CIFAR-10  | BiT-L ResNet152x4 | JFT-300M (private)    | **99.37%** |
| CIFAR-10  | BiT-M ResNet152x4 | ImageNet-21k (public) | 98.91%   |
| CIFAR-10  | BiT-S ResNet152x4 | ImageNet-1k (public)  | 97.51%   |
| CIFAR-100 | BiT-L ResNet152x4 | JFT-300M (private)    | **93.51%** |
| CIFAR-100 | BiT-M ResNet152x4 | ImageNet-21k (public) | 92.17%   |
| CIFAR-100 | BiT-S ResNet152x4 | ImageNet-1k (public)  | 86.21%   |

> ✅ BiT-M (ImageNet-21k) is publicly reproducible.
> Public code: [google-research/big_transfer](https://github.com/google-research/big_transfer)
> ⚠️ BiT-L requires JFT-300M, exclusive to Google.

---

### 2. Self-Supervised Pretraining via MoCo v3
**Paper**: *An Empirical Study of Training Self-Supervised Vision Transformers*
Chen et al. (2021) — arXiv:[2104.02057](https://arxiv.org/pdf/2104.02057)

**Method**: Trains a ViT backbone using **contrastive self-supervised learning** — no labels required during pretraining. Pretrained entirely on **public ImageNet-1k**, then fine-tuned on CIFAR with labels.

| Dataset   | Model      | Pretrain Data                  | Accuracy |
|-----------|------------|--------------------------------|----------|
| CIFAR-10  | ViT-L/16   | ImageNet-1k (self-supervised)  |   99.1%  |
| CIFAR-10  | ViT-B/16   | ImageNet-1k (self-supervised)  |   98.8%  |
| CIFAR-10  | ViT-H/14   | ImageNet-1k (self-supervised)  |   99.1%  |
| CIFAR-100 | ViT-L/16   | ImageNet-1k (self-supervised)  |   91.1%  |
| CIFAR-100 | ViT-B/16   | ImageNet-1k (self-supervised)  |   90.5%  |
| CIFAR-100 | ViT-H/14   | ImageNet-1k (self-supervised)  |   91.2%  |

> ✅ Beats supervised ImageNet-1k ViT by +4.7% on CIFAR-100 (86.4% → 91.1%).
> Public code: [facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3)

---

### 3. ASF-former (Adaptive Split-Fusion Transformer)
**Paper**: *Adaptive Split-Fusion Transformer*
Su et al. (2022) — arXiv:[2204.12196](https://arxiv.org/pdf/2204.12196)

**Method**: A CNN-Transformer hybrid that treats convolutional and attention branches differently with adaptive weights. Specifically, an ASF-former encoder equally splits feature channels into half to fit dual-path inputs. Then, the outputs of the dual path are fused with weights calculated from visual cues. Pretrained on **ImageNet-22k**, fine-tuned on CIFAR.

| Dataset   | Model         | Pretrain     | Accuracy  |
|-----------|---------------|--------------|-----------|
| CIFAR-10  | ASF-former-S  | ImageNet-22k |   98.7%   |
| CIFAR-10  | ASF-former-B  | ImageNet-22k |   98.8%   |
| CIFAR-100 | ASF-former-S  | ImageNet-22k |   90.4%   |
| CIFAR-100 | ASF-former-B  | ImageNet-22k |   91.0%   |

> ✅ Public code: [szx503045266/ASF-former](https://github.com/szx503045266/ASF-former)

---

## Part 2: Train from Scratch

These methods train entirely on CIFAR data with no external pretraining

---

### 1. AutoAugment
**Paper**: *AutoAugment: Learning Augmentation Policies from Data*
Cubuk et al. (2019) — arXiv:[1805.09501](https://arxiv.org/pdf/1805.09501) — **CVPR 2019**

**Method**: Use **Reinforcement Learning** to automatically search for improved data augmentation policies. In our implementation, we have designed a search space where a policy consists of many subpolicies, one of which is randomly chosen for each image in each mini-batch. A sub-policy consists of two operations, each operation being an image processing function such as translation, rotation, or shearing, and the probabilities and magnitudes with which the functions are applied. We use a search algorithm to find the best policy such that the neural network yields the highest validation accuracy on a target dataset. Once found, the policy is applied on top of a strong backbone (PyramidNet + ShakeDrop) during standard training.

| Dataset   | Backbone               | Accuracy   |
|-----------|------------------------|------------|
| CIFAR-10  | PyramidNet + ShakeDrop | **98.5%** |
| CIFAR-100 | PyramidNet + ShakeDrop | **89.3%** |

> ✅ Public code:
> [tensorflow/models/autoaugment](https://github.com/tensorflow/models/blob/master/research/autoaugment/README.md)

---

### 2. Fast AutoAugment
**Paper**: *Fast AutoAugment*
Lim et al. (2019) — arXiv:[1905.00397](https://arxiv.org/pdf/1905.00397) — **NeurIPS 2019**

**Method**: Use **Bayesian Optimization** to find effective augmentation policies via a more efficient search strategy based on density matching. In comparison to AutoAugment, the proposed algorithm speeds up the search time by orders of magnitude while achieves comparable performances on image recognition tasks with various models and datasets

| Dataset   | Backbone               | Accuracy   |
|-----------|------------------------|------------|
| CIFAR-10  | PyramidNet + ShakeDrop | **98.3%** |
| CIFAR-100 | PyramidNet + ShakeDrop | **88.3%** |


> ✅ Public code:
> [kakaobrain/fast-autoaugment](https://github.com/kakaobrain/fast-autoaugment)

---

### 3. Efficient Adaptive Ensembling
**Paper**: *Efficient Adaptive Ensembling for Image Classification*
Deufemia et al. (2022) — arXiv:[2206.07394](https://arxiv.org/pdf/2206.07394)

**Method**: First, trained two EfficientNet-b0 end-to-end models (known to be the architecture with the best overall accuracy/complexity trade-off for image classification) on disjoint subsets of data (i.e. bagging). Then, made an efficient adaptive ensemble by performing fine-tuning of a trainable combination layer

| Dataset   | Model                       | Accuracy    |
|-----------|-----------------------------|-------------|
| CIFAR-10  | EfficientNet-b0 × ensemble  | **99.612%** |
| CIFAR-100 | EfficientNet-b0 × ensemble  | **96.808%** |

> No public code repository cited. Result not independently reproduced.


### 4. Wide Residual Networks (WRN)
**Paper**: *Wide Residual Networks*
Zagoruyko & Komodakis (2016) — arXiv:[1605.07146](https://arxiv.org/pdf/1605.07146) — **BMVC 2016**

**Method**: Rather than stacking more layers, WRN **increases the width** of ResNet blocks by a widening factor k (up to 10×) while reducing depth. The key finding is that a 16-layer wide network can match or outperform a 1000-layer thin network, while training up to **8× faster**. Dropout is inserted between convolutional layers inside each residual block (not in the identity path) for regularization. Trained from scratch on CIFAR with standard flip + random crop augmentation only — no AutoAugment or external data.

| Dataset   | Model       | Params  | Accuracy   | Augmentation     |  Dropout  |
|-----------|-------------|---------|------------|------------------|-----------|
| CIFAR-10  | WRN-28-10   | 36.5M   | **95.83%** |      -           |     -     |
| CIFAR-10  | WRN-28-10   | 36.5M   | **96.00%** | flip/translation |     -     |
| CIFAR-10  | WRN-28-10   | 36.5M   | **96.11%** |      -           |    yes    | 
| CIFAR-10  | WRN-28-12   | 52.5M   | **95.67%** |      -           |     -     |
| CIFAR-10  | WRN-40-10   |   -     | **96.2%**  |      -           |    yes    |
| CIFAR-100 | WRN-28-10   | 36.5M   | **79.5%**  |      -           |     -     |
| CIFAR-100 | WRN-28-10   | 36.5M   | **80.75%** | flip/translation |     -     |
| CIFAR-100 | WRN-28-10   | 36.5M   | **81.15%** |      -           |    yes    |
| CIFAR-100 | WRN-28-12   | 52.5M   | **79.57%** |      -           |     -     |  
| CIFAR-100 | WRN-40-10   |   -     | **81.7%**  |      -           |    yes    |

> ✅ Public code:
> [szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)

---

### 5. Training ViT on Small-Scale Datasets
**Paper**: *How to Train Vision Transformer on Small-scale Datasets?*
Gani et al. (2022) — arXiv:[2210.07240](https://arxiv.org/pdf/2210.07240) — **BMVC 2022**

**Method**: A two-stage training framework that eliminates the need for large-scale pretraining for ViTs. **Stage 1 (Self-supervised)**: uses DINO-style local/global view prediction via a student-teacher setup to learn weight initialization directly from the small CIFAR dataset itself — capturing inductive biases without touching external data. **Stage 2 (Supervised)**: fine-tunes the same model on CIFAR using cross-entropy loss with standard augmentations (CutMix, MixUp, AutoAugment, label
 moothing, stochastic depth). Crucially, both stages use only the target CIFAR dataset.

| Dataset   | Model        | Params | Accuracy   |
|-----------|--------------|--------|------------|
| CIFAR-10  | ViT          | 2.8M   | **96.41%** |
| CIFAR-100 | ViT          | 2.8M   | **79.15%** |


> ✅ Public code:
> [hananshafi/vits-for-small-scale-datasets](https://github.com/hananshafi/vits-for-small-scale-datasets)



