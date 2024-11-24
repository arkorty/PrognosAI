from tabulate import tabulate
from colorama import Fore, Style
import torch
import random


def get_advice(vitals, prediction):
    """
    Generate advice based on vital signs and prediction.
    """
    advice = []
    if prediction == "High Risk":
        advice.append("Immediate medical attention is recommended.")

    if vitals["Heart Rate"] > 100:
        advice.append("Heart rate is high; consider resting and monitoring.")
    elif vitals["Heart Rate"] < 60:
        advice.append("Heart rate is low; consult a cardiologist if persistent.")

    if vitals["Body Temperature"] > 37.5:
        advice.append("High body temperature detected; monitor for fever.")
    elif vitals["Body Temperature"] < 36:
        advice.append("Low body temperature detected; check for hypothermia.")

    if vitals["Oxygen Saturation"] < 94:
        advice.append("Low oxygen saturation; consider oxygen therapy.")

    return " ".join(advice) if advice else "Vitals are within normal ranges."


def predict_random_samples(model, dataset, features, num_samples=10):
    """
    Predict risk categories for random samples from the dataset and display results with advice.

    Parameters:
    - model: Trained PyTorch model.
    - dataset: DataFrame containing the features.
    - features: List of feature column names.
    - num_samples: Number of random samples to predict.
    """
    # Check for GPU (ROCm or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Select random samples
    sampled_data = dataset.sample(num_samples)

    results = []
    with torch.no_grad():
        for _, row in sampled_data.iterrows():
            # Extract relevant features and ensure they are numeric
            data_point = row[features].astype(float).values
            vitals = row.to_dict()

            # Convert to tensor and move to GPU
            data_point_tensor = torch.tensor(data_point, dtype=torch.float32).to(device)
            output = model(data_point_tensor).squeeze().item()
            prediction = "High Risk" if output > 0.5 else "Low Risk"
            actual = row["Risk Category"]

            # Add colored status
            pred_color = Fore.RED if prediction == "High Risk" else Fore.GREEN
            actual_color = Fore.RED if actual == "High Risk" else Fore.GREEN
            advice = get_advice(vitals, prediction)

            # Store the results in a structured format for tabulation
            results.append(
                {
                    "Heart Rate": vitals["Heart Rate"],
                    "Resp Rate": vitals["Respiratory Rate"],
                    "Temp": vitals["Body Temperature"],
                    "Oxygen": vitals["Oxygen Saturation"],
                    "Prediction": f"{pred_color}{prediction}{Style.RESET_ALL}",
                    "Actual": f"{actual_color}{actual}{Style.RESET_ALL}",
                    "Advice": advice,
                }
            )

    # Define headers for the table
    headers = [
        "Heart Rate",
        "Resp Rate",
        "Temp",
        "Oxygen",
        "Prediction",
        "Actual",
        "Advice",
    ]

    # Convert results to list of lists for tabulate
    tabular_data = [
        [
            row["Heart Rate"],
            row["Resp Rate"],
            row["Temp"],
            row["Oxygen"],
            row["Prediction"],
            row["Actual"],
            row["Advice"],
        ]
        for row in results
    ]

    # Print the formatted table
    print(
        tabulate(
            tabular_data, headers=headers, tablefmt="fancy_grid", showindex="always"
        )
    )
