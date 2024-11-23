import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import Dataset

def load_and_preprocess_data(file_path, features, label):
    df = pd.read_csv(file_path)

    # Encode the labels
    label_encoder = LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])

    # Normalize the features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Split into train and test sets
    X = df[features].values
    y = df[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

class HealthDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
