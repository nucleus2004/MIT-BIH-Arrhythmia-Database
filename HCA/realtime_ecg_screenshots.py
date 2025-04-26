import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

def load_sample_ecg():
    try:
        data = pd.read_csv('data/mitbih_train.csv').values[1000:2000, :]
        return data[:, :-1]
    except:
        t = np.linspace(0, 10, 1000)
        ecg = np.zeros_like(t)
        for i in range(len(t)):
            # P wave
            ecg[i] += 0.25 * np.exp(-(t[i] % 1 - 0.2)**2 / 0.01)
            # QRS complex
            ecg[i] += np.exp(-(t[i] % 1 - 0.5)**2 / 0.004)
            # T wave
            ecg[i] += 0.3 * np.exp(-(t[i] % 1 - 0.7)**2 / 0.01)
        return ecg.reshape(-1, 187)

def create_monitor_screenshot(data, start_idx, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Style the plot
    ax.grid(True, color='#2C5545', alpha=0.3)
    ax.set_title('ECG Monitor', color='lime', pad=10, fontsize=14)
    ax.set_xlabel('Time (ms)', color='lime')
    ax.set_ylabel('Amplitude (mV)', color='lime')
    
    # Set tick colors
    ax.tick_params(colors='lime')
    for spine in ax.spines.values():
        spine.set_color('#2C5545')
    
    # Plot ECG line
    x = np.arange(200)
    y = data[start_idx:start_idx+200, 0]
    ax.plot(x, y, color='lime', linewidth=2)
    
    # Add vital signs
    hr = 75 + 5 * np.sin(start_idx/50)
    ax.text(0.02, 0.95, f'HR: {int(hr)} BPM', transform=ax.transAxes, color='lime', fontsize=12)
    current_time = datetime.now().strftime('%H:%M:%S')
    ax.text(0.85, 0.95, current_time, transform=ax.transAxes, color='lime', fontsize=12)
    
    # Set axis limits
    ax.set_xlim(0, 200)
    ax.set_ylim(-2, 2)
    
    # Save the figure
    plt.savefig(f'figures/{filename}.png', facecolor='black', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load data
    data = load_sample_ecg()
    
    # Create screenshots at different positions
    positions = [0, 200, 400]  # Different starting positions for variety
    for i, pos in enumerate(positions):
        create_monitor_screenshot(data, pos, f'ecg_monitor_sample_{i+1}')

if __name__ == '__main__':
    main() 