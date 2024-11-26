from tabulate import tabulate
from colorama import Fore, Style
from src.util.preprocess.pressure import preprocess_data
from src.model.ffnn import FFNNModel
from src.model.rnn import RNNModel
import torch
import random


def predict_pressure(model, data_path, samples=10):
    """
    Predict using the trained RNN model and display results in a tabular format.

    Args:
        model_path (str): Path to the saved RNN model.
        data_path (str): Path to the dataset CSV file.
        samples (int, optional): Number of random samples to display. Default: 10.

    Returns:
        None
    """
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    _, _, X_test_tensor, y_test_tensor, input_size = preprocess_data(data_path, device)

    # Initialize and load the model
    hidden_size = 64  # Same hidden size used during training
    rnn = RNNModel(input_size=input_size, hidden_size=hidden_size, output_size=1)
    rnn.load_state_dict(torch.load(model))
    rnn.to(device)
    rnn.eval()

    # Randomly sample from test data
    random_indices = random.sample(range(len(X_test_tensor)), samples)
    random_X = X_test_tensor[random_indices]
    random_y_true = y_test_tensor[random_indices]

    with torch.no_grad():
        random_y_pred = rnn(random_X).cpu().numpy()

    random_y_true = random_y_true.cpu().numpy()

    # Prepare and display the table
    table_data = []
    for i in range(samples):
        sample_id = random_indices[i]
        cpressure = random_y_true[i][0]  # Current pressure
        rpressure = random_y_pred[i][0]  # Recommended pressure
        dpressure = rpressure - cpressure  # Desired differential

        # Color coding for differential
        if dpressure > 0:
            dpressure_colored = f"{Fore.BLUE}{dpressure:.2f}{Style.RESET_ALL}"
        else:
            dpressure_colored = f"{Fore.RED}{dpressure:.2f}{Style.RESET_ALL}"

        table_data.append([sample_id, f"{cpressure:.2f}", dpressure_colored])

    headers = ["Sample ID", "Current Pressure", "Desired Differential"]
    print(
        tabulate(table_data, headers=headers, tablefmt="fancy_grid", showindex="always")
    )


def predict_risk(model, dataset, features, samples=10):
    """
    Predict risk categories for random samples from the dataset and display results.

    Parameters:
    - model: Trained PyTorch model.
    - dataset: DataFrame containing the features.
    - features: List of feature column names.
    - samples: Number of random samples to predict.
    """
    # Check for GPU (ROCm or CUDA)

    ffnn = FFNNModel(input_size=len(features))
    ffnn.load_state_dict(torch.load(model, weights_only=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")

    ffnn = ffnn.to(device)
    ffnn.eval()

    # Select random samples
    sampled_data = dataset.sample(samples)

    results = []
    with torch.no_grad():
        for _, row in sampled_data.iterrows():
            # Extract relevant features and ensure they are numeric
            data_point = row[features].astype(float).values
            vitals = row.to_dict()

            # Convert to tensor and move to GPU
            data_point_tensor = torch.tensor(data_point, dtype=torch.float32).to(device)
            output = ffnn(data_point_tensor).squeeze().item()
            prediction = "High Risk" if output > 0.5 else "Low Risk"
            actual = row["Risk Category"]

            # Determine the colors for prediction and actual values
            pred_color = Fore.RED if prediction == "High Risk" else Fore.GREEN
            actual_color = Fore.RED if actual == "High Risk" else Fore.GREEN

            # Store the results in a structured format for tabulation
            results.append(
                {
                    "Heart Rate": vitals["Heart Rate"],
                    "Resp Rate": vitals["Respiratory Rate"],
                    "Temp": vitals["Body Temperature"],
                    "Oxygen": vitals["Oxygen Saturation"],
                    "Prediction": f"{pred_color}{prediction}{Style.RESET_ALL}",
                    "Actual": f"{actual_color}{actual}{Style.RESET_ALL}",
                }
            )

    # Define headers for the table
    headers = ["Heart Rate", "Resp Rate", "Temp", "Oxygen", "Prediction", "Actual"]

    # Convert results to list of lists for tabulate
    table_data = [
        [
            row["Heart Rate"],
            row["Resp Rate"],
            row["Temp"],
            row["Oxygen"],
            row["Prediction"],
            row["Actual"],
        ]
        for row in results
    ]

    # Print the formatted table
    print(
        tabulate(table_data, headers=headers, tablefmt="fancy_grid", showindex="always")
    )
