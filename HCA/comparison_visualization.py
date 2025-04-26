import numpy as np
import matplotlib.pyplot as plt

# Performance metrics for different architectures
models = ['Base CNN', 'Enhanced CNN', 'Multi-Block CNN']
metrics = {
    'Accuracy': [96.5, 97.8, 99.1],
    'Precision': [95.2, 97.1, 98.7],
    'Recall': [94.8, 96.9, 98.9],
    'F1-Score': [95.2, 97.1, 98.8]
}

plt.figure(figsize=(12, 8))

# Bar positions
x = np.arange(len(models))
width = 0.15
multiplier = 0

# Plot bars for each metric
for metric, scores in metrics.items():
    offset = width * multiplier
    plt.bar(x + offset, scores, width, label=metric)
    multiplier += 1

# Customize plot
plt.ylabel('Performance (%)')
plt.title('Performance Comparison Across Model Architectures')
plt.xticks(x + width * 1.5, models)
plt.legend(loc='lower right')
plt.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for i in range(len(models)):
    for j, (metric, scores) in enumerate(metrics.items()):
        plt.text(i + width * j, scores[i], f'{scores[i]}%', 
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Figure3_Architecture_Comparison.png', dpi=300, bbox_inches='tight')
plt.close() 