from src.train import train_model
from src.evaluate import evaluate_model
from utils.data_preparation import HealthDataset
from torch.utils.data import DataLoader

# Configurations
DATA_PATH = "data/human_vital_signs_dataset_2024.csv"
FEATURES = [
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Age",
    "Weight (kg)",
    "Height (m)",
    "Derived_HRV",
    "Derived_Pulse_Pressure",
    "Derived_BMI",
    "Derived_MAP",
]
LABEL = "Risk Category"

# Train the model
model, X_test, y_test = train_model(DATA_PATH, FEATURES, LABEL)

# Prepare the test loader
test_dataset = HealthDataset(X_test, y_test)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Evaluate the model on the test set
acc, precision, recall, f1 = evaluate_model(model, test_loader)
print(
    f"Test Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}"
)
