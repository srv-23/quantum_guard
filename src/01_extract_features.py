import os
from utils import extract_metadata

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCAP_DIR = os.path.join(BASE_DIR, "captures")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def main():
    print("--- Step 1: Feature Extraction ---")
    
    # Define paths
    train_pcap = os.path.join(PCAP_DIR, "train.pcapng")
    test_pcap = os.path.join(PCAP_DIR, "test.pcapng")
    train_raw_csv = os.path.join(DATASET_DIR, "train_raw.csv")
    test_raw_csv = os.path.join(DATASET_DIR, "test_raw.csv")

    # Extract features from Train and Test PCAPs
    if os.path.exists(train_pcap):
        extract_metadata(train_pcap, train_raw_csv)
    else:
        print(f"Warning: {train_pcap} not found. Skipping extraction.")
    
    if os.path.exists(test_pcap):
        extract_metadata(test_pcap, test_raw_csv)
    else:
        print(f"Warning: {test_pcap} not found. Skipping extraction.")

    print("--- Extraction Complete ---")

if __name__ == "__main__":
    main()
