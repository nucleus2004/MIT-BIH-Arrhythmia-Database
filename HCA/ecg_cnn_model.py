import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Dense, Dropout, Flatten, 
                                   BatchNormalization, GaussianNoise, Input, Add,
                                   Multiply, GlobalAveragePooling1D, Reshape)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Set up paths
DATA_DIR = Path('data')
TRAIN_FILE = DATA_DIR / 'mitbih_train.csv'
TEST_FILE = DATA_DIR / 'mitbih_test.csv'

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE, header=None)
test_df = pd.read_csv(TEST_FILE, header=None)

# Prepare data
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print("\nClass weights:", class_weight_dict)

# Enhanced data augmentation function
def augment_ecg(signal, noise_level=0.05):
    # Ensure signal is 1D
    signal = signal.flatten()
    
    # Add random noise
    noise = np.random.normal(0, noise_level, signal.shape)
    augmented = signal + noise
    
    # Random scaling
    scale = np.random.uniform(0.9, 1.1)
    augmented = augmented * scale
    
    # Random time shift
    shift = np.random.randint(-5, 5)
    if shift > 0:
        augmented = np.pad(augmented[:-shift], (shift, 0), 'edge')
    elif shift < 0:
        augmented = np.pad(augmented[-shift:], (0, -shift), 'edge')
    
    # Random frequency shift
    freq_shift = np.random.uniform(-0.1, 0.1)
    t = np.arange(len(signal))
    augmented = augmented * np.exp(2j * np.pi * freq_shift * t)
    augmented = np.real(augmented)
    
    return augmented.reshape(-1, 1)

# Create augmented dataset for minority classes
def create_augmented_dataset(X, y, target_samples=20000):
    augmented_X = []
    augmented_y = []
    
    for class_idx in range(1, 5):  # Skip class 0 (majority class)
        class_indices = np.where(y == class_idx)[0]
        current_samples = len(class_indices)
        
        if current_samples < target_samples:
            num_augmentations = target_samples - current_samples
            for _ in range(num_augmentations):
                idx = np.random.choice(class_indices)
                augmented_signal = augment_ecg(X[idx])
                augmented_X.append(augmented_signal.flatten())  # Ensure 1D array
                augmented_y.append(class_idx)
    
    return np.array(augmented_X), np.array(augmented_y)

# Create augmented dataset
print("Creating augmented dataset...")
augmented_X, augmented_y = create_augmented_dataset(X_train, y_train)

# Combine original and augmented data
X_train_combined = np.vstack([X_train, augmented_X])
y_train_combined = np.hstack([y_train, augmented_y])

# Shuffle the combined dataset
indices = np.random.permutation(len(X_train_combined))
X_train_combined = X_train_combined[indices]
y_train_combined = y_train_combined[indices]

# Reshape data for CNN
X_train_combined = X_train_combined.reshape(X_train_combined.shape[0], X_train_combined.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert labels to one-hot encoding
y_train_combined = tf.keras.utils.to_categorical(y_train_combined)
y_test = tf.keras.utils.to_categorical(y_test)

print(f"\nTraining set shape after augmentation: {X_train_combined.shape}")

# Attention mechanism
def attention_block(input_tensor):
    attention = Conv1D(1, kernel_size=1, activation='sigmoid')(input_tensor)
    attention = Reshape((-1,))(attention)
    attention = tf.keras.layers.RepeatVector(input_tensor.shape[-1])(attention)
    attention = Reshape(input_tensor.shape[1:])(attention)
    return Multiply()([input_tensor, attention])

# Enhanced CNN Model with Residual Connections and Attention
def create_enhanced_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Initial Convolution
    x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    
    # Residual Block 1
    residual = x
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Residual Block 2 with Attention
    residual = Conv1D(128, kernel_size=1, padding='same')(x)  # 1x1 conv to match channels
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = attention_block(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Residual Block 3
    residual = Conv1D(256, kernel_size=1, padding='same')(x)  # 1x1 conv to match channels
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense Layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the enhanced model
print("Creating enhanced model...")
model = create_enhanced_model((X_train_combined.shape[1], 1), y_train_combined.shape[1])

# Custom learning rate schedule with warmup
initial_learning_rate = 0.001
warmup_steps = 1000
total_steps = 10000

def lr_schedule(step):
    if step < warmup_steps:
        return initial_learning_rate * (step / warmup_steps)
    return initial_learning_rate * tf.math.exp(-0.1 * (step - warmup_steps) / total_steps)

optimizer = Adam(learning_rate=lr_schedule(0))
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Enhanced callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    min_delta=0.0001
)

model_checkpoint = ModelCheckpoint(
    'best_model_enhanced.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=False
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    min_delta=0.0001
)

# Train the enhanced model
print("Training enhanced model...")
history = model.fit(
    X_train_combined, y_train_combined,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1,
    class_weight=class_weight_dict
)

# Save the final model
model.save('ecg_cnn_model_enhanced.h5')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history_enhanced.png')
plt.close()

# Evaluate the model
print("\nEvaluating enhanced model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes)) 