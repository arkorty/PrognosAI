import argparse
import pandas as pd
import torch
import sys
from src.train import train_ffnn
from src.train import train_rnn
from src.model.ffnn import FFNNModel
from src.predict import predict_risk
from src.predict import predict_pressure
from src.util.evaluate import evaluate_ffnn
from src.util.visualize import visualize_data
from src.util.visualize import visualize_ffnn

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
    parser.add_argument("--visualize", action="store_true", help="Visualize the dataset.")
    parser.add_argument("--train", type=str, help="Train the model.")
    parser.add_argument("--predict", type=str, help="Predict random samples.")
    parser.add_argument(
        "--save",
        type=str,
        required="--visualize" in sys.argv,
        help="Path to save directory.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required="--visualize" in sys.argv or "--train" in sys.argv or "--predict" in sys.argv,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required="--train" in sys.argv or "--predict" in sys.argv,
        help="Path to save/load the model.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--bsize", type=int, default=32, help="Batch size for training.")
    parser.add_argument(
        "--hsize",
        type=int,
        default=64,
        help="Hidden size for RNN model.",
    )
    parser.add_argument("--lrate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random samples for prediction.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.visualize:
        visualize_data(args.data, FEATURES, LABEL, args.save)
        return

    elif args.train:
        if "ffnn" in args.train:
            print("Training the model...")
            model, X_test, y_test = train_ffnn(
                data_path=args.data,
                features=FEATURES,
                label=LABEL,
                num_epochs=args.epochs,
                batch_size=args.bsize,
                learning_rate=args.lrate,
                model_save_path=args.model,
            )
            print("Model training complete.")

            conf_matrix, fpr, tpr, roc_auc = evaluate_ffnn(model, X_test, y_test)

            if args.save is not None:
                visualize_ffnn(conf_matrix, fpr, tpr, roc_auc, args.save)

        if "rnn" in args.train:
            train_rnn(
                data_path=args.data,
                model_path=args.model,
                num_epochs=args.epochs,
                batch_size=args.bsize,
                learning_rate=args.lrate,
            )

    elif args.predict:
        if "ffnn" in args.predict:
            predict_risk(args.model, df, FEATURES, args.samples)

        if args.predict and "rnn" in args.predict:
            predict_pressure(
                model=args.model,
                data_path=args.data,
                samples=args.samples,
            )
    else:
        print("Please specify --train, --predict, or --visualize.")


if __name__ == "__main__":
    main()
