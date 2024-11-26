from sklearn.model_selection import train_test_split
import pandas as pd
import torch


class HealthDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(data_path, features, label, test_size=0.2, val_size=0.1):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Preprocess features and labels
    X = data[features].values
    y = (
        (data[label] == "High Risk").astype(float).values
    )  # Binary encoding for risk category

    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Further split train+val into train and validation
    val_ratio = val_size / (1 - test_size)  # Adjust val size to remaining train+val set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
