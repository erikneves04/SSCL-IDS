import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(input_csv, output_csv):
    """
    Preprocess the dataset: Convert all columns to numeric where possible,
    encode categorical data, and classify rows based on the 'label' column.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the preprocessed CSV file.
    """
    # Load the dataset
    df = pd.read_csv(input_csv, header=None)
    
    # Define column names (adjust as needed based on the dataset)
    df.columns = [
        "timestamp", "id", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
        "service", "duration", "orig_bytes", "resp_bytes", "conn_state", 
        "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
        "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "label"
    ]
    
    # Create a LabelEncoder instance
    label_encoder = LabelEncoder()

    # Convert numerical columns to numeric (ignore non-numeric columns)
    numeric_columns = [
        "timestamp", "src_port", "dst_port", "duration", "orig_bytes", 
        "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", 
        "resp_pkts", "resp_ip_bytes"
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values in numerical columns with 0 (or use another method like median)
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Encode categorical columns
    categorical_columns = [
        "id", "src_ip", "dst_ip", "protocol", "service", "conn_state",
        "local_orig", "local_resp", "history"
    ]
    
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    
    # Encode the 'label' column (0 for benign, 1 for malicious)
    df['label'] = df['label'].apply(lambda x: 0 if x == '-' else 1)

    # Save the preprocessed DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed dataset saved to {output_csv}")

# Example usage
input_csv = "iot23-mirai-7-1.csv"  # Path to your input dataset
output_csv = "iot23-mirai-7-1-processed.csv"  # Path to save the processed dataset
preprocess_dataset(input_csv, output_csv)