import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, test_loader):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move data to the same device as the model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch).squeeze()
            preds = (outputs > 0.5).float()

            # Move results back to CPU for metrics
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return acc, precision, recall, f1
