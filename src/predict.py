import torch

def predict_new_data(model, new_data_point):
    model.eval()
    with torch.no_grad():
        new_data_point = torch.tensor(new_data_point, dtype=torch.float32)
        output = model(new_data_point).squeeze()
        return "High Risk" if output > 0.5 else "Low Risk"
