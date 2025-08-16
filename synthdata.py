import os
from datetime import timedelta

import pandas as pd
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer


def load_and_clean_data(file_path):
    """
    Load the CSV file and handle missing values.
    Interpolate to fill in NaNs where appropriate.
    """
    df = pd.read_csv(file_path)

    # Convert 'Time' to numeric if it's not already
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    # Handle missing values using interpolation
    df.interpolate(inplace=True, method="linear", limit_direction="both")

    return df


def process_synthetic_set(set_df):
    """
    Process a single synthetic set to handle duplicate times.

    Args:
        set_df (pandas.DataFrame): DataFrame for one synthetic set.

    Returns:
        pandas.DataFrame: Processed DataFrame with unique times and interpolated values.
    """
    # Sort by time
    set_df_sorted = set_df.sort_values("Time")

    # Remove duplicates, keeping the first occurrence
    set_df_unique = set_df_sorted.drop_duplicates(subset="Time", keep="first")

    # Identify all unique times in order
    unique_times = sorted(set(set_df_unique["Time"]))
    full_time_range = pd.DataFrame({"Time": unique_times})

    # Merge to identify gaps and interpolate
    merged_df = pd.merge(full_time_range, set_df_unique, on="Time", how="left")

    # Interpolate missing values for each vital sign column
    vital_signs = [
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
    for col in vital_signs:
        merged_df[col] = merged_df[col].interpolate()

    return merged_df


def round_numerical_values(df):
    """
    Rounds numerical columns to one decimal place.

    Args:
        df (pandas.DataFrame): DataFrame containing data.

    Returns:
        pandas.DataFrame: DataFrame with rounded numerical values.
    """
    vital_signs = [
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

    for col in vital_signs:
        if col in df.columns:
            df[col] = df[col].round(1)

    return df


# Main Execution
if __name__ == "__main__":
    # Parameters
    original_file_path = "Pig_Data_Master_Set.csv"  # Replace with your actual file path
    output_directory = "augmented_data"

    # Load and clean original data
    original_df = load_and_clean_data(original_file_path)

    # Detect and update metadata, setting both sequence key and time as a primary attribute
    metadata = Metadata.detect_from_dataframe(
        data=original_df, table_name="medical_data"
    )
    metadata.update_column(
        column_name="Set Number",
        sdtype="id",  # Or appropriate PII type if necessary
        table_name="medical_data",
    )
    metadata.set_sequence_key(column_name="Set Number")

    # Set 'Time' as a numerical column
    metadata.update_column(
        column_name="Time", sdtype="numerical", table_name="medical_data"
    )

    # Validate the updated metadata
    metadata.validate()

    # Initialize and train the synthesizer with all data
    synthesizer = PARSynthesizer(metadata)
    synthesizer.fit(original_df)

    # Calculate the original time interval (assuming regular intervals in original data)
    # Extract times from a single set to calculate the interval
    if not original_df["Set Number"].empty:
        first_set = original_df[
            original_df["Set Number"] == original_df["Set Number"].unique()[0]
        ]
        if len(first_set) >= 2:
            time_diff = first_set.iloc[1]["Time"] - first_set.iloc[0]["Time"]
            # Ensure time_diff is treated as a numeric interval
            original_time_interval = time_diff
        else:
            print(
                "Not enough data points to calculate time interval. Using default of 300 seconds (5 minutes)."
            )
            original_time_interval = 300  # Default interval in seconds
    else:
        print("No sets found in the data.")
        exit()

    # Directory to save synthetic data
    os.makedirs(output_directory, exist_ok=True)

    # Generate and process synthetic data for each original set
    unique_sets = original_df["Set Number"].unique()

    for set_num in unique_sets:
        print(f"Processing Set {set_num}...")

        # Generate synthetic data for this set
        num_sequences_per_set = 10  # Adjust as needed
        synthetic_data = synthesizer.sample(num_sequences=num_sequences_per_set)

        # Process to remove duplicates and interpolate
        processed_synthetic = process_synthetic_set(synthetic_data)

        # Round numerical values
        final_synthetic = round_numerical_values(processed_synthetic)

        # Save each synthetic dataset with unique naming
        for idx, (_, subset_df) in enumerate(final_synthetic.groupby("Set Number")):
            file_name = f"synthetic_data_set{set_num}_aug{idx}.csv"
            file_path = os.path.join(output_directory, file_name)

            # Ensure time is sorted before saving and maintain numeric time
            subset_df_sorted = subset_df.sort_values("Time")
            subset_df_sorted.to_csv(file_path, index=False)
            print(f"Saved {file_name} successfully.")

    print("Synthetic data generation completed for all sets.")
