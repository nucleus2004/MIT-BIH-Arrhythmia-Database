import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the datasets
print("\n=== MIT-BIH Training Dataset ===")
mitbih_train = pd.read_csv('mitbih_train.csv')
print(f"Shape: {mitbih_train.shape}")
print("\nFirst 5 rows:")
print(mitbih_train.head())

print("\n=== MIT-BIH Testing Dataset ===")
mitbih_test = pd.read_csv('mitbih_test.csv')
print(f"Shape: {mitbih_test.shape}")
print("\nFirst 5 rows:")
print(mitbih_test.head())

print("\n=== PTBDB Normal Dataset ===")
ptbdb_normal = pd.read_csv('ptbdb_normal.csv')
print(f"Shape: {ptbdb_normal.shape}")
print("\nFirst 5 rows:")
print(ptbdb_normal.head())

print("\n=== PTBDB Abnormal Dataset ===")
ptbdb_abnormal = pd.read_csv('ptbdb_abnormal.csv')
print(f"Shape: {ptbdb_abnormal.shape}")
print("\nFirst 5 rows:")
print(ptbdb_abnormal.head())

# Basic statistics
print("\n=== Dataset Statistics ===")
print("\nMIT-BIH Training:")
print(mitbih_train.describe())

print("\nPTBDB Normal:")
print(ptbdb_normal.describe())

print("\nPTBDB Abnormal:")
print(ptbdb_abnormal.describe())

# Plot some sample signals
plt.figure(figsize=(15, 10))

# Plot MIT-BIH samples
plt.subplot(2, 2, 1)
sample_mitbih = mitbih_train.iloc[0, :-1]  # Exclude the label column
plt.plot(sample_mitbih)
plt.title('MIT-BIH Sample Signal')
plt.grid(True)

# Plot PTBDB Normal sample
plt.subplot(2, 2, 2)
sample_normal = ptbdb_normal.iloc[0, :-1]  # Exclude the label column
plt.plot(sample_normal)
plt.title('PTBDB Normal Sample Signal')
plt.grid(True)

# Plot PTBDB Abnormal sample
plt.subplot(2, 2, 3)
sample_abnormal = ptbdb_abnormal.iloc[0, :-1]  # Exclude the label column
plt.plot(sample_abnormal)
plt.title('PTBDB Abnormal Sample Signal')
plt.grid(True)

plt.tight_layout()
plt.savefig('dataset_analysis.png')
plt.close()

print("\nAnalysis complete. Check 'dataset_analysis.png' for signal visualizations.") 