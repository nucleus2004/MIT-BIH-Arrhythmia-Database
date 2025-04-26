import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
DATA_DIR = Path('data')
TRAIN_FILE = DATA_DIR / 'mitbih_train.csv'
TEST_FILE = DATA_DIR / 'mitbih_test.csv'

# Load the datasets
print("Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE, header=None)
test_df = pd.read_csv(TEST_FILE, header=None)

# Print basic information
print("\nTraining set shape:", train_df.shape)
print("Test set shape:", test_df.shape)

# The last column is the class label
train_labels = train_df.iloc[:, -1]
test_labels = test_df.iloc[:, -1]

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=train_labels)
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.close()

# Print class counts
print("\nClass distribution in training set:")
print(train_labels.value_counts().sort_index())

# Plot sample ECG signals from each class
plt.figure(figsize=(15, 10))
for class_label in range(5):
    # Get first sample of each class
    sample = train_df[train_df.iloc[:, -1] == class_label].iloc[0, :-1]
    
    plt.subplot(2, 3, class_label + 1)
    plt.plot(sample)
    plt.title(f'Class {class_label} Sample')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('ecg_samples.png')
plt.close()

print("\nAnalysis complete. Check 'class_distribution.png' and 'ecg_samples.png' for visualizations.") 