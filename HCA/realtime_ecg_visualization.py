import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from datetime import datetime
import time

# Load a sample ECG signal from your dataset
def load_sample_ecg():
    try:
        # Try to load from MIT-BIH dataset
        data = pd.read_csv('data/mitbih_train.csv').values[1000:2000, :]  # Get 1000 samples
        return data[:, :-1]  # Remove the label column
    except:
        # If file not found, generate synthetic ECG-like data
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

# Create real-time ECG visualization
class ECGMonitor:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # Style the plot to look like a medical monitor
        self.ax.grid(True, color='#2C5545', alpha=0.3)
        self.ax.set_title('Real-time ECG Monitor', color='lime', pad=10, fontsize=14)
        self.ax.set_xlabel('Time (seconds)', color='lime')
        self.ax.set_ylabel('Amplitude (mV)', color='lime')
        
        # Set tick colors
        self.ax.tick_params(colors='lime')
        for spine in self.ax.spines.values():
            spine.set_color('#2C5545')
            
        # Initialize the line
        self.line, = self.ax.plot([], [], color='lime', linewidth=2)
        self.data = load_sample_ecg()
        self.current_idx = 0
        
        # Add vital signs display
        self.heart_rate = self.ax.text(0.02, 0.95, 'HR: 75 BPM', 
                                     transform=self.ax.transAxes, 
                                     color='lime', fontsize=12)
        self.time_text = self.ax.text(0.85, 0.95, '', 
                                    transform=self.ax.transAxes,
                                    color='lime', fontsize=12)
        
        # Set axis limits
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(-2, 2)
        
    def init(self):
        self.line.set_data([], [])
        return self.line,
    
    def update(self, frame):
        # Update ECG line
        x = np.arange(max(0, frame-200), frame)
        y = self.data[np.mod(x, len(self.data)), 0]  # Loop through data
        self.line.set_data(x, y)
        
        # Update time
        current_time = datetime.now().strftime('%H:%M:%S')
        self.time_text.set_text(current_time)
        
        # Simulate varying heart rate
        hr = 75 + 5 * np.sin(frame/50)
        self.heart_rate.set_text(f'HR: {int(hr)} BPM')
        
        return self.line, self.heart_rate, self.time_text

    def show(self):
        anim = FuncAnimation(self.fig, self.update, init_func=self.init,
                           frames=1000, interval=40, blit=True)
        plt.show()

if __name__ == '__main__':
    monitor = ECGMonitor()
    monitor.show() 