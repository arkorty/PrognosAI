import argparse
import pandas as pd
import torch
from models.risk_predictor import RiskPredictor
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_random_samples
from src.utils.preprocessing import load_and_preprocess_data

# Feature and label definitions
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


def main():
    parser = argparse.ArgumentParser(description="Healthcare Risk Prediction")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument(
        "--predict", action="store_true", help="Predict random samples."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to save/load the model."
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples for prediction.",
    )
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.data)

    if args.train:
        print("Training the model...")
        model, X_test, y_test = train_model(
            data_path=args.data,
            features=FEATURES,
            label=LABEL,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_save_path=args.model,
        )
        print("Model training complete.")

        evaluate_model(model, X_test, y_test)

    elif args.predict:
        print("Loading the model for prediction...")
        model = RiskPredictor(input_size=len(FEATURES))
        model.load_state_dict(torch.load(args.model))

        print("Predicting random samples from the dataset...")
        predictions = predict_random_samples(model, df, FEATURES, args.num_samples)

    else:
        print("Please specify either --train or --predict.")


if __name__ == "__main__":
    main()
