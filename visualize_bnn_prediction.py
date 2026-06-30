# visualize_bnn_predictions.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

# ---------- Config ----------
# Paths to your saved npy files
result_path = './results/resnet32_bayesian/cifar10/'  # adjust path if needed

PROBS_PATH  = os.path.join(result_path, 'probs_cifar_mc.npy')   # adjust filename
LABELS_PATH = os.path.join(result_path, 'cifar_test_labels_mc.npy') # adjust filename

# CIFAR-10 data root 
DATA_ROOT   = './data/cifar10'

# output path for visualizations
OUTPUT_DIR  = os.path.join(result_path, 'visualizations')                 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of classes in CIFAR-10
NUM_CLASSES = 10
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ---------- Load model outputs ----------
probs_mc = np.load(PROBS_PATH)      # [num_mc, N_test, num_classes]
labels  = np.load(LABELS_PATH)      # [N_test]

num_mc, N_test, num_classes = probs_mc.shape
assert num_classes == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {num_classes}"

print(f"Loaded probs_mc with shape {probs_mc.shape}")
print(f"Loaded labels with shape {labels.shape}")

# Mean and variance over MC samples
mean_probs = probs_mc.mean(axis=0)        # [N_test, num_classes]
var_probs  = probs_mc.var(axis=0)         # [N_test, num_classes]

# Predicted label from mean_probs
preds = mean_probs.argmax(axis=1)        # [N_test]

# ---------- Load CIFAR-10 test dataset ----------
transform = transforms.ToTensor()
test_ds   = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
assert len(test_ds) == N_test, f"Dataset length {len(test_ds)} != N_test {N_test}"

# ---------- Visualization function ----------
def visualize_sample(idx, save=True):
    """
    Show CIFAR-10 image, true/pred labels, mean predictive distribution,
    variance, save the figure to disk.
    """
    img_tensor, true_label = test_ds[idx]
    mean_p = mean_probs[idx]
    var_p  = var_probs[idx]
    pred_label = preds[idx]

    img = img_tensor.numpy()
    img = np.transpose(img, (1, 2, 0))  # [H, W, C]

    plt.figure(figsize=(12, 4))

    # Image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(
        f"Index {idx}\n"
        f"True: {CLASS_NAMES[true_label]} ({true_label})\n"
        f"Pred: {CLASS_NAMES[pred_label]} ({pred_label})\n"
        f"p_max = {mean_p[pred_label]:.2f}"
    )

    # Mean predictive distribution
    plt.subplot(1, 3, 2)
    classes = np.arange(NUM_CLASSES)
    plt.bar(classes, mean_p)
    plt.xticks(classes, CLASS_NAMES, rotation=45, ha='right')
    plt.ylabel("Mean predicted probability")
    plt.ylim(0, 1)
    plt.title("Predictive mean over MC samples")

    # Variance
    plt.subplot(1, 3, 3)
    plt.bar(classes, var_p)
    plt.xticks(classes, CLASS_NAMES, rotation=45, ha='right')
    plt.ylabel("Variance across MC samples")
    plt.title("Per-class predictive variance")

    plt.tight_layout()

    if save:
        filename = os.path.join(OUTPUT_DIR, f"sample_{idx:05d}.png")
        plt.savefig(filename, dpi=150)
        print(f"Saved visualization to {filename}")

    plt.close()  # close the figure to avoid memory issues

# ---------- Main ----------
if __name__ == '__main__':
    # Example: visualize and save a few samples
    for idx in [0, 1, 2, 100, 999]:
        visualize_sample(idx, save=True)