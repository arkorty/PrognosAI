from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from models.risk_predictor import RiskPredictor
from utils.data_preparation import HealthDataset, load_and_preprocess_data


def train_model(
    data_path, features, label, num_epochs=20, batch_size=32, learning_rate=0.001
):
    # Check for GPU (ROCm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (via ROCm)")
    else:
        print("No GPU detected. Using CPU.")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(
        data_path, features, label
    )
    train_dataset = HealthDataset(X_train, y_train)
    val_dataset = HealthDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and move to GPU
    model = RiskPredictor(input_size=len(features)).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss}"
        )

    return model, X_test, y_test
