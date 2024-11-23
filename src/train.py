from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.risk_predictor import RiskPredictor
from utils.data_preparation import HealthDataset, load_and_preprocess_data

def train_model(data_path, features, label, num_epochs=20, batch_size=32, learning_rate=0.001):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, features, label)

    train_dataset = HealthDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = RiskPredictor(input_size=len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return model
