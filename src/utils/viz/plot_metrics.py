import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def plot_metrics_from_log(log_file_path):
    epoch_nums = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    pattern = re.compile(
        r"Epoch (\d+)/\d+: Train Loss=([\d.]+), Train Acc=([\d.]+), Val Loss=([\d.]+), Val Acc=([\d.]+)"
    )

    with open(log_file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_acc = float(match.group(3))
                val_loss = float(match.group(4))
                val_acc = float(match.group(5))
                
                epoch_nums.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

    if not epoch_nums:
        print("No matching data found in log.")
        return

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_nums, train_losses, label="Train Loss")
    plt.plot(epoch_nums, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch_nums, train_accs, label="Train Accuracy")
    plt.plot(epoch_nums, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_from_npy(probs_file, labels_file, threshold=0.5, class_names=None):

    all_probs = np.load(probs_file)  # shape: (num_samples, num_classes)
    all_labels = np.load(labels_file)  # shape: (num_samples,)

    # For multiclass, convert probabilities to predicted classes
    # If binary classification with probabilities on one class, use threshold
    predicted_classes = np.argmax(all_probs, axis=1)

    if all_labels.ndim > 1 and all_labels.shape[1] > 1:  # assuming one-hot encoded labels if needed
        true_classes = np.argmax(all_labels, axis=1)
    else:
        true_classes = all_labels.astype(int)

    cm = confusion_matrix(true_classes, predicted_classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve_from_npy(probs_file, labels_file, class_names=None):
    all_probs = np.load(probs_file)  # shape: (num_samples, num_classes)
    all_labels = np.load(labels_file)  # shape: (num_samples,)

    # Binarize labels if they are not binary (one-hot encoding or discrete)
    if all_labels.ndim == 1 or all_labels.shape[1] == 1:
        true_bin = label_binarize(all_labels, classes=np.arange(all_probs.shape[1]))
    else:
        true_bin = all_labels

    n_classes = all_probs.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        label = class_names[i] if class_names else f"Class {i}"
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"ROC curve of {label} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


