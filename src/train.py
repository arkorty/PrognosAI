from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model.ffnn import FFNNModel
from src.model.rnn import RNNModel
from src.util.preprocess.risk import HealthDataset, load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def train_rnn(
    data_path,
    model_path="model.pth",
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    hidden_size=64,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vital_signs_df = pd.read_csv(data_path)

    vital_signs_df = vital_signs_df.rename(
        columns={
            "Heart Rate": "HeartRate",
            "Respiratory Rate": "RespiratoryRate",
            "Body Temperature": "BodyTemperature",
            "Oxygen Saturation": "SpO2",
            "Systolic Blood Pressure": "BloodPressureSystolic",
            "Diastolic Blood Pressure": "BloodPressureDiastolic",
        }
    )

    if "VentilatorPressure" not in vital_signs_df.columns:
        np.random.seed(42)
        vital_signs_df["VentilatorPressure"] = (
            60
            + 0.1 * vital_signs_df["HeartRate"]
            + 0.3 * vital_signs_df["RespiratoryRate"]
            + 0.2 * vital_signs_df["SpO2"]
            + np.random.normal(0, 5, len(vital_signs_df))
        )

    features = [
        "HeartRate",
        "RespiratoryRate",
        "BodyTemperature",
        "SpO2",
        "BloodPressureSystolic",
        "BloodPressureDiastolic",
        "Derived_BMI",
    ]
    target = "VentilatorPressure"

    X = vital_signs_df[features].values
    y = vital_signs_df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[2]
    output_size = 1
    model = RNNModel(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def train_ffnn(
    data_path,
    features,
    label,
    num_epochs=20,
    batch_size=32,
    learning_rate=0.001,
    model_save_path="model/risk.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (via ROCm)")
    else:
        print("No GPU detected. Using CPU.")

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_path, features, label)
    train_dataset = HealthDataset(X_train, y_train)
    val_dataset = HealthDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FFNNModel(input_size=len(features)).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, X_test, y_test
