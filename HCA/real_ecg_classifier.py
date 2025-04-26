import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

def load_ecg_data(data_path):
    """
    Load ECG data from CSV files
    Expected format: Each file should have ECG signals in columns and a label column
    """
    X = []
    Y = []
    
    # Check if data_path is a directory or file
    if os.path.isdir(data_path):
        # Load all CSV files in the directory
        for file in os.listdir(data_path):
            if file.endswith('.csv'):
                file_path = os.path.join(data_path, file)
                try:
                    df = pd.read_csv(file_path)
                    # Assuming last column is the label
                    signals = df.iloc[:, :-1].values
                    labels = df.iloc[:, -1].values
                    X.extend(signals)
                    Y.extend(labels)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
    else:
        # Load single file
        try:
            df = pd.read_csv(data_path)
            signals = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
            X.extend(signals)
            Y.extend(labels)
        except Exception as e:
            print(f"Error loading file: {str(e)}")
    
    return np.array(X), np.array(Y)

def train_ecg_classifier(data_path):
    """
    Train ECG classifier on real data
    """
    # Load data
    print("Loading ECG data...")
    X, Y = load_ecg_data(data_path)
    
    if len(X) == 0:
        print("No data loaded. Please check your data files.")
        return
    
    print(f"Loaded {len(X)} ECG samples")
    
    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define and train the neural network
    print("\nTraining model...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        verbose=True
    )
    
    # Train the model
    model.fit(X_train_scaled, Y_train)
    
    # Make predictions
    Y_pred = model.predict(X_test_scaled)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))
    
    # Print accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    return model, scaler

if __name__ == "__main__":
    # Example usage
    data_path = "sample_ecg_data.csv"
    model, scaler = train_ecg_classifier(data_path) 