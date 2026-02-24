import pyshark
import csv
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

def extract_metadata(pcap_path, output_csv_path):
    """Extracts TLS metadata from a .pcapng file to a CSV."""
    if not os.path.exists(pcap_path):
        print(f"Error: File {pcap_path} not found.")
        return

    print(f"Extracting features from {pcap_path}...")
    cap = pyshark.FileCapture(pcap_path, display_filter="tls")
    
    prev_time = None
    flow_start = None

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["length", "inter_arrival", "record_type", "flow_duration"])

        for pkt in cap:
            try:
                ts = float(pkt.sniff_timestamp)
                length = int(pkt.length)

                if flow_start is None:
                    flow_start = ts
                    prev_time = ts

                inter = ts - prev_time if prev_time else 0
                prev_time = ts

                # Try to get record content type, default to 0 if not present or fails
                try:
                    record = int(pkt.tls.record_content_type)
                except (AttributeError, ValueError):
                    record = 0
                
                flow = ts - flow_start

                writer.writerow([length, inter, record, flow])
            except AttributeError:
                pass
            except Exception as e:
                continue
    
    cap.close()
    print(f"Saved raw metadata to {output_csv_path}")

def preprocess_dataset(input_csv, output_csv, scaler_path=None, fit_scaler=False, inject_anomalies=False):
    """Preprocesses the raw CSV: One-Hot Encoding, Scaling, and Labeling."""
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV {input_csv} not found.")
        return None

    df = pd.read_csv(input_csv)

    if df.empty:
        print(f"Warning: {input_csv} is empty.")
        return None

    # One-hot encode TLS record type (ensure consistent columns in real scenarios)
    # We expect these common TLS record types: 20 (ChangeCipherSpec), 21 (Alert), 22 (Handshake), 23 (Application Data)
    expected_records = [20, 21, 22, 23]
    for r_type in expected_records:
        col_name = f"record_type_{r_type}"
        df[col_name] = (df["record_type"] == r_type).astype(int)
    
    # Drop original record_type column
    df.drop(columns=["record_type"], inplace=True)
    
    # Ensure all expected columns exist
    expected_cols = ["length", "inter_arrival", "flow_duration", 
                     "record_type_20", "record_type_21", "record_type_22", "record_type_23"]
    
    # Keep only expected columns (adds 0s if missing, reorders)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Scaling
    if fit_scaler:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
    else:
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            scaled_data = scaler.transform(df)
        else:
            print("Warning: No scaler found. Skipping scaling.")
            scaled_data = df.values # Fallback

    processed = pd.DataFrame(scaled_data, columns=expected_cols)

    # Label generation (Synthetic for this demo - unless we have real labels)
    # In a real scenario, you would merge with a label file or use folder names
    processed["label"] = 0
    if inject_anomalies:
        # Mark random 10% as anomalous (label 1) for demonstration
        anomaly_indices = np.random.choice(len(processed), size=int(0.1 * len(processed)), replace=False)
        processed.loc[anomaly_indices, "label"] = 1

    processed.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")
    return processed
