import numpy as np
from typing import Dict, Text, List
import matplotlib.pyplot as plt


def plot_metrics(history: Dict[Text, int], metrics: List[str]):
    """
    Parameters
    ----------
    history: Dict[Text,int]
        Input dictionary containing model metrics from tf.keras model
    metrics: List[str]
        Input list of metrics for plotting
    """
    n_cols = 2
    n_rows = np.ceil(len(metrics) / n_cols)
    epochs = [i for i in range(0, len(history["val_loss"]))]

    fig = plt.figure(figsize=(8.0 * n_cols, 5.5 * n_rows))
    fig.subplots_adjust(hspace=0.20)
    for idx, name in enumerate(metrics):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.plot(epochs, history[metrics[idx]], marker="o", label="train")
        ax.plot(epochs, history["val_" + metrics[idx]], marker="o", label="validation")
        ax.set_xlabel("Epoch")
        plt.ylabel("Top-K Accuracy")
        ax.set_title(f"{name} vs epoch", fontsize=12)
        ax.legend()
    plt.suptitle("Top-K Accuracy vs Epoch", fontsize=24, y=0.95)
