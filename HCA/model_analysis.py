import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import time
import psutil
import os

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

def plot_confusion_matrix(y_true, y_pred, model_name, class_names):
    """Plot confusion matrix for a model"""
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {model_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {model_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png')
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    start_time = time.time()
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    inference_time = time.time() - start_time
    
    # Get memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get classification report
    report = classification_report(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1),
        output_dict=True
    )
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'loss': loss,
        'inference_time': inference_time,
        'memory_usage': memory_usage,
        'classification_report': report,
        'predictions': y_pred
    }

def compare_models(models_dict, X_test, y_test, class_names):
    """Compare multiple models and generate comparison plots"""
    results = []
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(model, X_test, y_test, model_name)
        results.append(result)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, result['predictions'], model_name, class_names)
    
    # Create comparison table
    comparison_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Accuracy': r['accuracy'],
        'Loss': r['loss'],
        'Inference Time (s)': r['inference_time'],
        'Memory Usage (MB)': r['memory_usage']
    } for r in results])
    
    # Save comparison table
    comparison_df.to_csv('model_comparison.csv', index=False)
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=comparison_df)
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    return results

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Define class names
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    
    # Load all models
    models = {
        'Base Model': tf.keras.models.load_model('ecg_cnn_model_base.h5'),
        'Enhanced Model': tf.keras.models.load_model('ecg_cnn_model_enhanced.h5'),
        'Best Model': tf.keras.models.load_model('best_model_enhanced.h5'),
        'Final Model': tf.keras.models.load_model('ecg_cnn_model_enhanced.h5')
    }
    
    # Compare models
    results = compare_models(models, X_test, y_test, class_names)
    
    # Print detailed comparison
    print("\nDetailed Model Comparison:")
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Loss: {result['loss']:.4f}")
        print(f"Inference Time: {result['inference_time']:.2f}s")
        print(f"Memory Usage: {result['memory_usage']:.2f}MB")
        print("\nClassification Report:")
        print(pd.DataFrame(result['classification_report']).transpose())

if __name__ == "__main__":
    main() 