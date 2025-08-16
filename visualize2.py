import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd


def load_original_data(file_path):
    """
    Load the original dataset from a CSV file.

    Args:
        file_path (str): Path to the original CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df


def load_synthetic_data(synthetic_dir, target_set):
    """
    Load all synthetic datasets from a specified directory that correspond to the target set.

    Args:
        synthetic_dir (str): Directory containing synthetic data files.
        target_set (int): The original set number for which to load synthetic data.

    Returns:
        list of pandas.DataFrame: List of DataFrames, each representing a synthetic dataset for the target set.
    """
    # Find all synthetic files related to this set
    synthetic_files = glob(
        os.path.join(synthetic_dir, f"synthetic_data_set{target_set}_aug*.csv")
    )

    synthetic_dfs = []
    for file in synthetic_files:
        df = pd.read_csv(file)
        synthetic_dfs.append(df)

    return synthetic_dfs


def create_line_plots(original_df, synthetic_dfs, target_set):
    """
    Create line plots comparing the original and synthetic data for a specific set.

    Args:
        original_df (pandas.DataFrame): Original dataset.
        synthetic_dfs (list of pandas.DataFrame): List of synthetic datasets for the target set.
        target_set (int): The set number being visualized.
    """
    # Extract columns to plot
    columns_to_plot = [
        "SpO2",
        "Palpebral",
        "Jaw Tone",
        "Anesthetic Rate",
        "HR",
        "SAP",
        "RR",
        "CO2",
        "Temperature",
    ]

    # Create figure and subplots
    fig, axes = plt.subplots(
        len(columns_to_plot), 1, figsize=(15, len(columns_to_plot) * 4)
    )

    if len(columns_to_plot) == 1:
        axes = [axes]  # Ensure axes is a list for consistent handling

    # Filter original data for the target set
    original_set = original_df[original_df["Set Number"] == target_set]

    # Process each column for plotting
    for i, (col, ax) in enumerate(zip(columns_to_plot, axes)):
        # Plot original data
        ax.plot(
            original_set["Time"],
            original_set[col],
            label=f"Original Set {target_set}",
            color="blue",
            linewidth=2,
        )

        # Plot synthetic data
        if len(synthetic_dfs) > 0:
            for idx, df in enumerate(synthetic_dfs):
                ax.plot(
                    df["Time"],
                    df[col],
                    label=f"Synthetic Augmentation {idx + 1}",
                    color=plt.cm.viridis(idx / len(synthetic_dfs)),
                    alpha=0.7,
                    linewidth=1,
                )

        # Customize plot
        ax.set_title(f"{col} Over Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    original_file_path = "Pig_Data_Master_Set.csv"  # Replace with your actual file path
    synthetic_directory = "augmented_data"  # Directory containing synthetic data

    target_set = 2  # Choose the set you want to visualize (e.g., Set 1)

    # Load data
    original_data = load_original_data(original_file_path)
    synthetic_datasets = load_synthetic_data(synthetic_directory, target_set)

    # Create plots
    create_line_plots(original_data, synthetic_datasets, target_set)
