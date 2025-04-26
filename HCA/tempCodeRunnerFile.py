import numpy as np
import wfdb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

def extract_features(signal, fs=360):
    """Extract features from ECG signal"""
    # Basic statistical features
    features = []
    for channel in range(signal.shape[1]):
        channel_signal = signal[:, channel]
        features.extend([
            np.mean(channel_signal),      # Mean
            np.std(channel_signal),       # Standard deviation
            np.max(channel_signal),       # Maximum value
            np.min(channel_signal),       # Minimum value
            np.percentile(channel_signal, 75) - np.percentile(channel_signal, 25)  # IQR
        ])
    return np.array(features)

def load_and_process_record(record_name, data_dir):
    """Load and process a single record"""
    try:
        # Read the signal and annotations
        record_path = os.path.join(data_dir, record_name)
        record = wfdb.rdrecord(record_path)
        annot = wfdb.rdann(record_path, 'atr')
        
        # Get signal data
        signal = record.p_signal
        
        # Process in windows of 10 seconds (3600 samples at 360 Hz)
        window_size = 3600
        features_list = []
        labels_list = []
        
        # Process each window
        for i in range(0, len(signal) - window_size, window_size):
            window = signal[i:i+window_size]
            features = extract_features(window)
            
            # Find annotations that fall within this window
            ann_indices = np.where((annot.sample >= i) & (annot.sample < i+window_size))[0]
            
            if len(ann_indices) > 0:
                # Get the most common beat type in this window
                window_annotations = [annot.symbol[idx] for idx in ann_indices]
                label = max(set(window_annotations), key=window_annotations.count)
                
                features_list.append(features)
                labels_list.append(label)
        
        if len(features_list) > 0:
            return np.array(features_list), np.array(labels_list)
        else:
            return None, None
            
    except Exception as e:
        print(f"Error processing record {record_name}: {str(e)}")
        return None, None

# Directory containing the MIT-BIH database
data_dir = "mit-bih-arrhythmia-database-1.0.0"

# List of all records from MIT-BIH Arrhythmia Database
records = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

print(f"Total records to process: {len(records)}")

# Collect data from all records
X_all = []
y_all = []

print("Loading and processing records...")
processed_count = 0
error_count = 0

for record in records:
    try:
        X, y = load_and_process_record(record, data_dir)
        if X is not None and y is not None:
            X_all.append(X)
            y_all.append(y)
            processed_count += 1
            print(f"Successfully processed record {record} ({processed_count}/{len(records)})")
        else:
            error_count += 1
            print(f"Failed to process record {record} - No valid data segments found")
    except Exception as e:
        error_count += 1
        print(f"Error processing record {record}: {str(e)}")

print(f"\nProcessing complete:")
print(f"Successfully processed: {processed_count} records")
print(f"Failed to process: {error_count} records")

if len(X_all) == 0:
    print("No records were successfully processed. Please check the data directory and record names.")
    exit()

# Combine all data
X = np.vstack(X_all)
y = np.concatenate(y_all)

# Convert string labels to numeric using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(f"\nInitial class distribution:")
for class_name, count in zip(le.classes_, np.bincount(y)):
    print(f"{class_name}: {count}")

# Remove classes with too few samples (less than 10 samples)
MIN_SAMPLES_PER_CLASS = 10
class_counts = np.bincount(y)
valid_classes = np.where(class_counts >= MIN_SAMPLES_PER_CLASS)[0]
mask = np.isin(y, valid_classes)

# Filter data and labels
X = X[mask]
y_filtered = y[mask]

# Create new label mapping for remaining classes
remaining_classes = le.classes_[valid_classes]
new_le = LabelEncoder()
y = new_le.fit_transform(le.inverse_transform(y_filtered))

print(f"\nAfter removing underrepresented classes:")
print(f"Total samples: {len(X)}")
print(f"Remaining classes: {new_le.classes_}")
print(f"Class distribution:")
for class_name, count in zip(new_le.classes_, np.bincount(y)):
    print(f"{class_name}: {count}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train SVM Model with class weights
print("\nTraining SVM model...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight=class_weight_dict,
    random_state=42
)
svm_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nSVM Model Accuracy: {accuracy * 100:.2f}%")

# Create a new figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(15, 10), facecolor='white')
gs = GridSpec(2, 2, figure=fig)
fig.suptitle('MIT-BIH Arrhythmia Classification Analysis', fontsize=16, y=0.95)

# Plot 1: Initial Class Distribution
ax1 = fig.add_subplot(gs[0, 0])
initial_classes = le.classes_
initial_counts = np.bincount(y, minlength=len(le.classes_))
colors = plt.cm.Set3(np.linspace(0, 1, len(initial_classes)))
bars1 = ax1.bar(range(len(initial_classes)), initial_counts, color=colors)
ax1.set_xticks(range(len(initial_classes)))
ax1.set_xticklabels(initial_classes, rotation=45)
ax1.set_title('Initial Class Distribution')
ax1.set_ylabel('Number of Samples')
# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

# Plot 2: Final Class Distribution
ax2 = fig.add_subplot(gs[0, 1])
final_classes = new_le.classes_
final_counts = np.bincount(y)
colors = plt.cm.Set3(np.linspace(0, 1, len(final_classes)))
bars2 = ax2.bar(range(len(final_classes)), final_counts, color=colors)
ax2.set_xticks(range(len(final_classes)))
ax2.set_xticklabels(final_classes, rotation=45)
ax2.set_title('Final Class Distribution\n(After Filtering)')
ax2.set_ylabel('Number of Samples')
# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

# Plot 3: Confusion Matrix
ax3 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=new_le.classes_,
            yticklabels=new_le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Plot 4: Model Performance Metrics
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Calculate per-class metrics
class_report = classification_report(y_test, y_pred, target_names=new_le.classes_, output_dict=True)
metrics_text = f"""
Model Performance Metrics:

• Overall Accuracy: {accuracy * 100:.2f}%

• Records Processed: {processed_count}/{len(records)}
• Total Samples: {len(X)}
• Training Samples: {len(X_train)}
• Testing Samples: {len(X_test)}

• Number of Classes: {len(new_le.classes_)}

Per-Class Performance:
"""
for class_name in new_le.classes_:
    metrics_text += f"\n• {class_name}:"
    metrics_text += f" Precision={class_report[class_name]['precision']:.2f}"
    metrics_text += f", Recall={class_report[class_name]['recall']:.2f}"
    metrics_text += f", F1={class_report[class_name]['f1-score']:.2f}"

plt.text(0, 1, metrics_text, fontsize=10, va='top')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save the results
plt.savefig('ecg_analysis_results.png', dpi=300, bbox_inches='tight', facecolor='white')

# Print detailed classification report to console
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=new_le.classes_))