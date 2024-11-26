import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tabulate import tabulate
from colorama import Fore, Style


def preprocess_data(data_path, device):
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_test_tensor, y_test_tensor, X_train.shape[2]
