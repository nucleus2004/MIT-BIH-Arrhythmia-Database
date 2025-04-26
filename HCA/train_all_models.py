import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from pathlib import Path
import time

# Set up paths
DATA_DIR = Path('data')
TRAIN_FILE = DATA_DIR / 'mitbih_train.csv'
TEST_FILE = DATA_DIR / 'mitbih_test.csv'

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE, header=None)
    test_df = pd.read_csv(TEST_FILE, header=None)
    
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    
    # Reshape for CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    return X_train, y_train, X_test, y_test

def create_base_model(input_shape, num_classes):
    """Create the base CNN model"""
    model = Sequential([
        # First Convolutional Layer
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second Convolutional Layer
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Third Convolutional Layer
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_enhanced_model(input_shape, num_classes):
    """Create the enhanced CNN model"""
    model = Sequential([
        # First Convolutional Block
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape, 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Second Convolutional Block
        Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Third Convolutional Block
        Conv1D(256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Fourth Convolutional Block
        Conv1D(512, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_base_model(X_train, y_train, X_test, y_test):
    """Train and save the base model"""
    print("\nTraining Base Model...")
    model = create_base_model((X_train.shape[1], 1), y_train.shape[1])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(patience=10)],
        verbose=1
    )
    
    model.save('ecg_cnn_model_base.h5')
    return model, history

def train_enhanced_model(X_train, y_train, X_test, y_test):
    """Train and save the enhanced model"""
    print("\nTraining Enhanced Model...")
    model = create_enhanced_model((X_train.shape[1], 1), y_train.shape[1])
    
    # Custom learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model_enhanced.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    model.save('ecg_cnn_model_enhanced.h5')
    return model, history

def main():
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Train base model
    base_model, base_history = train_base_model(X_train, y_train, X_test, y_test)
    
    # Train enhanced model
    enhanced_model, enhanced_history = train_enhanced_model(X_train, y_train, X_test, y_test)
    
    print("\nTraining completed. Models saved as:")
    print("1. Base Model: ecg_cnn_model_base.h5")
    print("2. Enhanced Model: ecg_cnn_model_enhanced.h5")
    print("3. Best Model: best_model_enhanced.h5")

if __name__ == "__main__":
    main() 