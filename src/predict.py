import torch


def predict_new_data(model, new_data_point):
    # Check for GPU (ROCm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Move new data to GPU
        new_data_point = torch.tensor(new_data_point, dtype=torch.float32).to(device)
        output = model(new_data_point).squeeze()
        return "High Risk" if output > 0.5 else "Low Risk"
