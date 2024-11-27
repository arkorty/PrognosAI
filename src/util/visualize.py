import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def visualize_ffnn(conf_matrix, fpr, tpr, roc_auc, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Greys",
        xticklabels=["Low", "High"],
        yticklabels=["Low", "High"],
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Confusion matrix saved to {filename}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="lightgray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    filename = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"ROC curve saved to {filename}")


def summarize_dataset(data, label_column, output_dir):
    print("Dataset Overview:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())

    plt.figure(figsize=(8, 6))
    data[label_column].value_counts().plot(kind="bar", color=["#8c8c8c", "#bfbfbf", "#d9d9d9"])
    plt.title("Class Distribution")
    plt.xlabel("Risk Category")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()


def plot_feature_distributions(data, features, output_dir):
    num_features = len(features)
    n_cols = 3
    n_rows = (num_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.histplot(data[feature], kde=True, ax=axes[i], color="#808080")
        axes[i].set_title(f"Distribution of {feature}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Frequency")

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
    plt.close()


def plot_correlation_matrix(data, features, output_dir):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data[features].corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="Greys",
        cbar=True,
        square=True,
        linecolor="gray",
        linewidths=0.5,
    )
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()


def pairplot_features(data, features, label_column, output_dir):
    pairplot = sns.pairplot(data[features + [label_column]], hue=label_column, palette="gray")
    pairplot.savefig(os.path.join(output_dir, "pairplot_features.png"))
    plt.close()


def check_missing_values(data, output_dir):
    missing_values = data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
    if missing_values.any():
        plt.figure(figsize=(10, 6))
        missing_values[missing_values > 0].plot(kind="bar", color="#a6a6a6")
        plt.title("Missing Values Count by Column")
        plt.xlabel("Columns")
        plt.ylabel("Missing Count")
        plt.savefig(os.path.join(output_dir, "missing_values.png"))
        plt.close()
    else:
        print("No missing values found!")


def visualize_data(data_path, features, label_column, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(data_path)

    summarize_dataset(data, label_column, output_dir)
    plot_feature_distributions(data, features, output_dir)
    plot_correlation_matrix(data, features, output_dir)
    check_missing_values(data, output_dir)
