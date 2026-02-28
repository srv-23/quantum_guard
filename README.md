# Quantum Guard: TLS Anomaly Detection

A research-grade machine learning pipeline for detecting anomalies in TLS traffic using flow-based metadata (no payload inspection). This project implements and compares **Supervised Learning (Random Forest)** and **Unsupervised Learning (Isolation Forest)** approaches.

## 🚀 Features

- **Privacy-Preserving**: Uses only packet metadata (packet lengths, inter-arrival times, TLS record types) without decrypting traffic.
- **Micro-Flow Extraction**: Parses raw PCAP files into flow-based statistical features using `PyShark` (tshark wrapper).
- **Strict Data Separation**: Dedicated split logic ensures zero leakage between training, validation, and testing sets.
  - **Isolation Forest**: Trained strictly on benign traffic.
  - **Random Forest**: Trained on labeled mixed traffic.
- **Advanced Evaluation**: 
  - Automated threshold calibration for unsupervised models.
  - ROC, Precision-Recall, and Confusion Matrix visualizations.
  - Generates comprehensive metrics (Accuracy, F1, Precision, Recall, AUC).

## 📂 Project Structure

```
quantum_guard/
├── dataset/                  # Generated CSV datasets
│   ├── processed/            # Final split datasets (train_if, train_rf, test_set)
│   ├── train_benign.csv      # Raw features from benign PCAP
│   └── test_mixed.csv        # Raw features from mixed PCAP
├── docker/                   # Docker configuration for deployment
├── models/                   # Saved models (.pkl) and scalers
├── outputs/                  # Evaluation plots (ROC curves, confusion matrices)
├── pcap/                     # Input PCAP files
│   ├── train_benign.pcap     # Pure benign traffic for baseline
│   └── test_mixed.pcap       # Mixed benign/anomalous traffic
├── src/                      # Source code
│   ├── 1_extract_features.py       # PCAP parsing & feature extraction
│   ├── 2_create_split_datasets.py  # Data splitting & leakage prevention
│   ├── 3_train_models.py           # Model training & optimization
│   └── 4_evaluate_pipeline.py      # Final evaluation & plotting
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🛠 Prerequisites

- **Python 3.8+**
- **Wireshark / TShark**: This project uses `PyShark`, which requires `tshark` to be installed and available in your system PATH.
  - [Download Wireshark](https://www.wireshark.org/download.html)

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quantum_guard.git
   cd quantum_guard
   ```

2. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage Pipeline

Run the scripts in numerical order to execute the full pipeline.

### 1. Extract Features
Parses PCAP files from the `pcap/` directory and extracts flow metadata into CSV format.
```bash
python src/1_extract_features.py
```
*Input*: `pcap/train_benign.pcap`, `pcap/test_mixed.pcap`  
*Output*: `dataset/train_benign.csv`, `dataset/test_mixed.csv`

### 2. Create Datasets
Splits the raw CSV data into dedicated training and testing sets to ensure fair comparison and prevent data leakage.
```bash
python src/2_create_split_datasets.py
```
*Output*: 
- `dataset/processed/train_if.csv` (100% Benign for Unsupervised)
- `dataset/processed/train_rf.csv` (Mixed for Supervised)
- `dataset/processed/test_set.csv` (Hold-out test set)

### 3. Train Models
Trains the Random Forest classifier and optimizes the Isolation Forest (using Grid Search and Robust Scaling).
```bash
python src/3_train_models.py
```
*Output*: Saved models in `models/*.pkl`

### 4. Evaluate Pipeline
Evaluates both models on the unseen `test_set.csv`, calculates metrics, and generates performance plots.
```bash
python src/4_evaluate_pipeline.py
```
*Output*: Charts in `outputs/` folder (`roc_curve.png`, `confusion_matrix.png`) and console metrics.

## 📊 Methodology

### Isolation Forest (Unsupervised)
- **Training Data**: Strictly benign traffic.
- **Optimization**: Uses `GridSearchCV` to find optimal contamination and max_features.
- **Scaling**: Applied `RobustScaler` to handle outliers in flow features effectively.
- **Threshold**: Dynamically calibrated on a validation set to maximize F1 score.

### Random Forest (Supervised)
- **Training Data**: Labeled mixed traffic (Benign=0, Malicious=1).
- **Labeling Logic**: Flows with high `handshake_ratio` (>0.15) or any `alert` packets are labeled as anomalous based on TLS fingerprinting research.
- **Performance**: Typically achieves near-perfect separation on synthetic/clean datasets.

## 📈 Results
Check the `outputs/` folder for visual results.
- **ROC Curve**: Compares the trade-off between True Positive Rate and False Positive Rate.
- **Confusion Matrix**: Shows precise counts of True Benign, False Benign, True Anomaly, and False Anomaly.