import numpy as np
import matplotlib.pyplot as plt

# Simulated training history
epochs = range(1, 101)
training_acc = [0.89 + 0.1 * (1 - np.exp(-epoch/20)) + np.random.normal(0, 0.01) for epoch in epochs]
val_acc = [0.88 + 0.11 * (1 - np.exp(-epoch/25)) + np.random.normal(0, 0.01) for epoch in epochs]
training_loss = [0.4 * np.exp(-epoch/30) + 0.1 + np.random.normal(0, 0.01) for epoch in epochs]
val_loss = [0.45 * np.exp(-epoch/35) + 0.11 + np.random.normal(0, 0.01) for epoch in epochs]

plt.figure(figsize=(15, 6))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, training_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Model Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, training_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Model Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('Figure2_Training_Performance.png', dpi=300, bbox_inches='tight')
plt.close() 