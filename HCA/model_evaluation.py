import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from pathlib import Path

# Set up paths
DATA_DIR = Path('data')
TEST_FILE = DATA_DIR / 'mitbih_test.csv'
MODEL_PATH = 'ecg_cnn_model.h5'

# Load the test dataset
print("Loading test dataset...")
test_df = pd.read_csv(TEST_FILE, header=None)
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Reshape data for CNN
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load the trained model
print("Loading trained model...")
model = load_model(MODEL_PATH)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Find misclassified examples
misclassified_indices = np.where(y_test != y_pred_classes)[0]

# Plot some misclassified examples
plt.figure(figsize=(15, 10))
for i, idx in enumerate(misclassified_indices[:6]):
    plt.subplot(2, 3, i + 1)
    plt.plot(X_test[idx].flatten())
    plt.title(f'True: {y_test[idx]}, Pred: {y_pred_classes[idx]}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('misclassified_examples.png')
plt.close()

# Calculate and plot class-wise accuracy
class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
plt.figure(figsize=(10, 6))
plt.bar(range(len(class_accuracy)), class_accuracy)
plt.title('Class-wise Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.xticks(range(len(class_accuracy)))
plt.savefig('class_accuracy.png')
plt.close()

print("\nEvaluation complete. Check the following files for visualizations:")
print("- confusion_matrix.png: Shows the confusion matrix")
print("- misclassified_examples.png: Shows some misclassified ECG signals")
print("- class_accuracy.png: Shows accuracy per class") 