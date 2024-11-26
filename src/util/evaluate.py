import os
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_ffnn(model, X_test, y_test):
    """
    Evaluate the model using test data (X_test and y_test) and return evaluation results and plotting data.

    Args:
        model: Trained PyTorch model.
        X_test: Test features (tensor or NumPy array).
        y_test: Test labels (tensor or NumPy array).

    Returns:
        Tuple containing:
            - metrics: Dictionary with accuracy, precision, recall, and f1 score.
            - conf_matrix: Confusion matrix.
            - roc_data: Tuple of (fpr, tpr, roc_auc).
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (via ROCm)")
    else:
        print("No GPU detected. Using CPU.")

    model = model.to(device)
    model.eval()

    # Convert data to tensors if necessary
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)

    # Move data to the same device as the model
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Forward pass to get predictions
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        preds = (outputs > 0.5).float()

    # Convert predictions and labels back to CPU for metric calculation
    predictions = preds.cpu().numpy()
    true_labels = y_test.cpu().numpy()

    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    print(f"Model Evaluation Metrics:")
    print(f"  Accuracy:    {acc * 100:.2f}%")
    print(f"  Precision:   {precision * 100:.2f}%")
    print(f"  Recall:      {recall * 100:.2f}%")
    print(f"  F1 Score:    {f1 * 100:.2f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, outputs.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    return conf_matrix, fpr, tpr, roc_auc