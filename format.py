import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
data_path = "data/processed/preproc_dataset.csv"  # Path to your final preprocessed CSV
df = pd.read_csv(data_path)

# Step 1: Encode categorical variables
# Label encode 'class' (target) and 'wlan.ra' (receiver address)
label_encoders = {}

# Label encode 'class' (target variable)
class_encoder = LabelEncoder()
df["class"] = class_encoder.fit_transform(df["class"])
label_encoders["class"] = class_encoder

# Label encode 'wlan.ra' (receiver address)
ra_encoder = LabelEncoder()
df["wlan.ra"] = ra_encoder.fit_transform(df["wlan.ra"])
label_encoders["wlan.ra"] = ra_encoder

# Step 2: Normalize/Standardize numerical features
# Identify numerical columns
numerical_columns = ["frame.time_delta_displayed", "wlan.duration"]
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# Step 3: Create sequences
def create_sequences(data, seq_len=10):
    sequences, labels = [], []
    for i in range(len(data) - seq_len + 1):
        seq = data.iloc[i : i + seq_len].drop(columns=["class"]).values
        label = data.iloc[i + seq_len - 1][
            "class"
        ]  # Use label of the last packet in the sequence
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# Set sequence length
seq_len = 10
X, y = create_sequences(df, seq_len=seq_len)

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Save processed data for training
np.save("data/processed/train_data.npy", X_train)
np.save("data/processed/train_labels.npy", y_train)
np.save("data/processed/test_data.npy", X_test)
np.save("data/processed/test_labels.npy", y_test)

print("Preprocessed data saved for training.")
