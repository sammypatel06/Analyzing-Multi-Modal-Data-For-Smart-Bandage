import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def predict_future_values(input_csv_path, output_csv_path, steps_to_predict=10):
    # Load the pre-trained model
    loaded_model = tf.keras.models.load_model("surgical_prediction_model.h5")

    # Read input data and drop "Set Number"
    new_data = pd.read_csv(input_csv_path)
    new_data = new_data.drop(["Set Number"], axis=1)

    # Preprocess: Handle missing values and normalize
    imputer = KNNImputer(n_neighbors=5)
    new_data_imputed = pd.DataFrame(
        imputer.fit_transform(new_data), columns=new_data.columns
    )
    scaler = StandardScaler()
    scaled_new_data = scaler.fit_transform(new_data_imputed)

    # Prepare input sequences
    sequence_length = 10  # Must match the one used during training
    last_sequence = scaled_new_data[-sequence_length:]

    # Generate future predictions
    predictions = []
    for _ in range(steps_to_predict):
        next_values = loaded_model.predict(np.array([last_sequence]))[0]
        predictions.append(next_values)

        # Update sequence by rolling and appending new values
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = next_values

    # Inverse transform predictions to original scale
    predicted_values = scaler.inverse_transform(np.array(predictions))

    # Create DataFrame with appropriate column names
    predicted_df = pd.DataFrame(predicted_values, columns=new_data.columns)

    # Generate future time values starting from the maximum time in input data plus one 5-minute interval (300 seconds)
    last_time = new_data_imputed["Time"].max()
    future_times = [int(last_time) + (i + 1) * 300 for i in range(steps_to_predict)]
    predicted_df["Time"] = future_times

    # Round Time to nearest integer and other columns to one decimal place
    predicted_df["Time"] = predicted_df["Time"].round().astype(int)
    for col in predicted_df.columns:
        if col != "Time":
            predicted_df[col] = predicted_df[col].round(1)

    # Save predictions with correct column names (excluding "Set Number")
    predicted_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def main():
    input_csv = "augmented_data/synthetic_data_set10_aug0.csv"
    output_csv = "predicted_future_values.csv"
    predict_future_values(input_csv, output_csv, steps_to_predict=10)


if __name__ == "__main__":
    main()
