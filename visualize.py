import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd


def visualize_set(set_number, synthetic_dir="augmented_data"):
    """
    Visualize original and synthetic data for a specific set number.

    Parameters:
        set_number (int): The set number to visualize (e.g., 1).
        synthetic_dir (str): Directory containing synthetic data files.
    """
    # Load original dataset
    original_df = pd.read_csv(
        "Pig_Data_Master_Set.csv"
    )  # Replace with your original CSV file path

    # Filter original data for the specified set number
    original_set = original_df[original_df["Set Number"] == set_number]

    # Find all synthetic files related to this set
    synthetic_files = glob(
        os.path.join(synthetic_dir, f"synthetic_data_set{set_number}_aug*.csv")
    )

    # Load and combine synthetic data
    if synthetic_files:
        synthetic_dfs = []
        for file in synthetic_files:
            df = pd.read_csv(file)
            synthetic_dfs.append(df)

        synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)

        # Combine original and synthetic data
        combined_df = pd.concat([original_set, synthetic_df], ignore_index=True)
    else:
        print(f"No synthetic data found for Set {set_number}")
        return

    # Prepare the figure with subplots
    plt.figure(figsize=(15, 20))

    # List of vital signs to plot
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

    for i, column in enumerate(columns_to_plot):
        plt.subplot(5, 2, i + 1)

        # Plot original data
        plt.plot(
            original_set["Time"], original_set[column], label="Original", color="blue"
        )

        # Plot synthetic data
        for idx, file in enumerate(synthetic_files):
            df = pd.read_csv(file)
            plt.plot(
                df["Time"],
                df[column],
                label=f"Synthetic {idx + 1}",
                color=plt.cm.viridis(idx / len(synthetic_files)),
                alpha=0.7,
                linewidth=1,
            )

        plt.title(f"{column} Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel(column)
        plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage:
# Replace 'your_dataset.csv' with your actual original dataset path
visualize_set(set_number=2)  # Change the set number as needed
