# ECG Classification System

This project implements deep learning models for ECG signal classification using the MIT-BIH Arrhythmia Database.

## Project Structure

- `base_model.py`: Basic CNN model implementation
- `ecg_cnn_model.py`: Enhanced CNN model with attention mechanisms
- `train_all_models.py`: Training pipeline for both base and enhanced models
- `model_analysis.py`: Model evaluation and comparison tools
- `realtime_ecg_visualization.py`: Real-time ECG monitoring visualization
- `preprocessing_visualization.py`: ECG signal preprocessing functions
- `analyze_datasets.py`: Dataset analysis and visualization

## Requirements

See `requirements.txt` for all dependencies. Main requirements:
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
