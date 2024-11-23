from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_new_data
from utils.data_preparation import HealthDataset, load_and_preprocess_data
from torch.utils.data import DataLoader

# Configurations
DATA_PATH = "data/human_vital_signs_dataset_2024.csv"
FEATURES = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Systolic Blood Pressure", "Diastolic Blood Pressure",
    "Age", "Weight (kg)", "Height (m)", "Derived_HRV", 
    "Derived_Pulse_Pressure", "Derived_BMI", "Derived_MAP"
]
LABEL = "Risk Category"

# Train the model
model = train_model(DATA_PATH, FEATURES, LABEL)

# Load test data
_, X_test, _, y_test = load_and_preprocess_data(DATA_PATH, FEATURES, LABEL)
test_dataset = HealthDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model
acc, precision, recall, f1 = evaluate_model(model, test_loader)
print(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Predict on new data
new_data_point = [0.5, 0.3, 0.6, 0.9, 0.7, 0.6, 0.4, 0.5, 0.6, 0.2, 0.5, 0.4, 0.6]  # Example
print(predict_new_data(model, new_data_point))
