"""
Project Structure and File Dependencies

1. Data Processing Chain:
-----------------------
dataset_analysis.py
    ↓
real_ecg_classifier.py
    ↓
base_model.py
    ↓
ecg_cnn_model.py

2. Training Pipeline:
-------------------
train_all_models.py
    ↓
base_model.py
    ↓
ecg_cnn_model.py
    ↓
tempCodeRunnerFile.py (SVM implementation)

3. Evaluation Chain:
------------------
model_evaluation.py
    ↓
model_analysis.py
    ↓
IEEE_Documentation.md

File Purposes:
-------------
1. dataset_analysis.py:
   - Initial data exploration
   - Class distribution analysis
   - Signal visualization
   - Data quality checks

2. real_ecg_classifier.py:
   - Basic MLP classifier
   - Data loading utilities
   - Initial preprocessing
   - Basic evaluation

3. base_model.py:
   - Basic CNN architecture
   - 3 convolutional layers
   - Simple training setup
   - Baseline performance

4. ecg_cnn_model.py:
   - Enhanced CNN architecture
   - Data augmentation
   - Advanced training features
   - Best performance model

5. train_all_models.py:
   - Training pipeline
   - Model checkpointing
   - Training history
   - Model saving

6. model_evaluation.py:
   - Detailed metrics
   - Error analysis
   - Performance visualization
   - Class-wise accuracy

7. model_analysis.py:
   - Model comparison
   - Memory analysis
   - Speed benchmarks
   - Comparative metrics

8. tempCodeRunnerFile.py:
   - SVM implementation
   - Feature extraction
   - MIT-BIH processing
   - Alternative approach

9. IEEE_Documentation.md:
   - Project documentation
   - Results summary
   - Methodology
   - Future work

Dependencies:
------------
- numpy
- pandas
- tensorflow
- sklearn
- matplotlib
- seaborn
- wfdb (for MIT-BIH database)
- psutil (for memory analysis)

Data Flow:
---------
Raw Data → Preprocessing → Training → Evaluation → Documentation

Models Implemented:
-----------------
1. Basic MLP (real_ecg_classifier.py)
2. Basic CNN (base_model.py)
3. Enhanced CNN (ecg_cnn_model.py)
4. SVM (tempCodeRunnerFile.py)

Performance Metrics Tracked:
-------------------------
- Accuracy
- Loss
- Memory Usage
- Inference Time
- Class-wise Performance
""" 