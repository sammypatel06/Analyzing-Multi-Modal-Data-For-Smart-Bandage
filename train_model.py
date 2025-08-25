import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Set CUDA environment variables
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"] = (
    os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["CUDA_HOME"] + "/lib64"
)


def main():
    # Read synthetic data
    synthetic_path = "augmented_data"
    synthetic_files = [
        f
        for f in os.listdir(synthetic_path)
        if f.startswith("synthetic_data_set") and f.endswith(".csv")
    ]
    synthetic_dfs = []
    for file in synthetic_files:
        df = pd.read_csv(os.path.join(synthetic_path, file))
        synthetic_dfs.append(df)
    synthetic_data = pd.concat(synthetic_dfs, ignore_index=True)

    # Read real data
    real_data = pd.read_csv("Pig_Data_Master_Set.csv")

    # Combine all data
    all_sets = pd.concat([synthetic_data, real_data], ignore_index=True)

    # Identify unique sets
    unique_sets = all_sets["Set Number"].unique()

    # Randomly split into training and testing (60-40)
    train_ratio = 0.6
    test_ratio = 0.4

    np.random.seed(42)  # For reproducibility
    train_sets = np.random.choice(
        unique_sets, size=int(len(unique_sets) * train_ratio), replace=False
    )
    remaining_sets = [s for s in unique_sets if s not in train_sets]
    test_sets = np.random.choice(
        remaining_sets, size=int(len(unique_sets) * test_ratio), replace=False
    )

    # Assign data to training and testing sets
    train_data = all_sets[all_sets["Set Number"].isin(train_sets)]
    test_data = all_sets[all_sets["Set Number"].isin(test_sets)]

    # Remove Set Number column
    train_data = train_data.drop("Set Number", axis=1)
    test_data = test_data.drop("Set Number", axis=1)

    # Impute missing values in training data
    imputer = KNNImputer(n_neighbors=5)
    train_data_imputed = pd.DataFrame(
        imputer.fit_transform(train_data), columns=train_data.columns
    )

    # Normalize data
    scaler = StandardScaler()
    train_data_normalized = scaler.fit_transform(train_data_imputed)
    test_data_normalized = scaler.transform(
        test_data.dropna()
    )  # Drop rows with NaN in test data

    # Create sequences for LSTM
    sequence_length = 10  # Adjust as needed

    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            x_sample = data[i : i + seq_length]
            y_sample = data[i + seq_length]
            X.append(x_sample)
            y.append(y_sample)
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data_normalized, sequence_length)
    X_test, y_test = create_sequences(test_data_normalized, sequence_length)

    # Build and train LSTM model
    model = Sequential()
    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(sequence_length, train_data_normalized.shape[1]),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(train_data_normalized.shape[1]))

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Use GPU if available
    with tf.device("/gpu:0"):
        history = model.fit(
            X_train, y_train, epochs=50, batch_size=32, validation_split=0.2
        )

    # Save the trained model
    model.save("surgical_prediction_model.h5")


if __name__ == "__main__":
    main()
