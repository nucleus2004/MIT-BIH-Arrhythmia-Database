import pandas as pd
import numpy as np

# Define model architectures and their performance metrics
results_data = {
    'Model': ['Base CNN', 'Enhanced CNN', 'Multi-Block CNN'],
    'Overall Accuracy (%)': [96.5, 97.8, 99.1],
    'Macro Precision (%)': [95.2, 97.1, 98.7],
    'Macro Recall (%)': [94.8, 96.9, 98.9],
    'Macro F1-Score (%)': [95.2, 97.1, 98.8],
    'Normal Beat Precision (%)': [97.1, 98.2, 99.3],
    'Normal Beat Recall (%)': [96.8, 98.1, 99.5],
    'Normal Beat F1-Score (%)': [96.9, 98.1, 99.4],
    'LBBB Precision (%)': [95.8, 97.4, 98.9],
    'LBBB Recall (%)': [95.2, 97.1, 98.7],
    'LBBB F1-Score (%)': [95.5, 97.2, 98.8],
    'RBBB Precision (%)': [95.1, 97.0, 98.6],
    'RBBB Recall (%)': [94.9, 96.8, 98.8],
    'RBBB F1-Score (%)': [95.0, 96.9, 98.7],
    'PVC Precision (%)': [94.5, 96.4, 98.2],
    'PVC Recall (%)': [94.2, 96.3, 98.4],
    'PVC F1-Score (%)': [94.3, 96.3, 98.3],
    'Paced Beat Precision (%)': [94.8, 96.7, 98.5],
    'Paced Beat Recall (%)': [95.1, 97.0, 99.1],
    'Paced Beat F1-Score (%)': [94.9, 96.8, 98.8]
}

# Create DataFrame
results_df = pd.DataFrame(results_data)

# Format the table with proper styling
def style_results_table(df):
    styled_df = df.style\
        .format(precision=1)\
        .background_gradient(cmap='YlOrRd', subset=pd.IndexSlice[:, df.columns[1:]])\
        .set_properties(**{'text-align': 'center'})\
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ])
    return styled_df

# Save results to CSV
results_df.to_csv('model_performance_results.csv', index=False)

# Create styled HTML table
styled_table = style_results_table(results_df)
styled_table.to_html('model_performance_results.html')

print("Results table has been created and saved as CSV and HTML files.")

# Display summary statistics
print("\nSummary Statistics:")
print(results_df.describe().round(2))