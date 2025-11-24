# User Guide

## Introduction
This guide provides step-by-step instructions to set up, run, and experiment with ML-for-Encrypted-Network-Classification.

## Installation Steps
1. Clone the repo:
   ```
   git clone https://github.com/Ezana-Fekadu/ML-for-Encrypted-Network-Classification.git
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Dataset Setup
- Refer to [Datasets-table.md](../Datasets-table.md) for supported datasets.
- Download and place datasets in the recommended directory.

## Usage Guide
Run the main script:
```
python main.py --input <path_to_data> --model <model_name> [options]
```

### Example:
```
python main.py --input data/encrypted_packets.csv --model RandomForest
```

## Configuration Options
- `--input` : Path to your dataset (required)
- `--model` : ML model to use (RandomForest, SVM, etc.)
- Additional options specified via `--help`

## Understanding Results
- Output includes classification accuracy, confusion matrix, and (if enabled) encrypted traffic analysis reports.

## Troubleshooting
- Ensure Python dependencies are installed.
- For dataset errors, verify file paths and data format.

## Getting Help
Open a GitHub issue or contact the maintainer.

---