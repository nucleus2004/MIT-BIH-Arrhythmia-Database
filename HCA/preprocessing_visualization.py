import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

# Create a sample ECG signal
def create_sample_ecg(n_points=500):
    t = np.linspace(0, 10, n_points)
    # Simulate baseline wander
    baseline = 0.5 * np.sin(2 * np.pi * 0.05 * t)
    # Create ECG-like peaks
    ecg = signal.gausspulse(t - 5, fc=2)
    # Add power line interference
    powerline = 0.2 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise
    # Add random noise
    noise = np.random.normal(0, 0.1, n_points)
    # Combine all components
    raw_signal = ecg + baseline + powerline + noise
    return t, raw_signal

# Preprocessing steps
def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def remove_baseline(signal, t):
    return signal - np.sin(2 * np.pi * 0.05 * t)

def remove_powerline(signal, t):
    # Remove 50/60 Hz interference
    powerline = 0.2 * np.sin(2 * np.pi * 50 * t)
    return signal - powerline

def wavelet_denoise(signal):
    # Wavelet denoising
    coeffs = pywt.wavedec(signal, 'db4', level=2)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, 'db4')

def augment_signal(signal):
    # Add small random noise
    noise = np.random.normal(0, 0.05, len(signal))
    # Scale amplitude
    scale = np.random.uniform(0.9, 1.1)
    # Time shift
    shift = np.random.randint(-5, 5)
    shifted = np.roll(signal, shift)
    return scale * (shifted + noise)

# Create visualization
plt.figure(figsize=(15, 12))

# Generate sample data
t, raw_signal = create_sample_ecg()

# Plot each step
steps = [
    ('Raw ECG Signal', raw_signal),
    ('Normalized Signal (Eq. 1)', normalize_signal(raw_signal)),
    ('Baseline Wander Removed (Eq. 2)', remove_baseline(raw_signal, t)),
    ('Power Line Interference Removed', remove_powerline(remove_baseline(raw_signal, t), t)),
    ('Wavelet Denoised', wavelet_denoise(remove_powerline(remove_baseline(raw_signal, t), t))),
    ('Augmented Signal', augment_signal(wavelet_denoise(remove_powerline(remove_baseline(raw_signal, t), t))))
]

for idx, (title, data) in enumerate(steps, 1):
    plt.subplot(len(steps), 1, idx)
    plt.plot(t, data, 'b-')
    plt.title(f'Step {idx}: {title}')
    plt.grid(True)
    if idx < len(steps):
        plt.xticks([])
    else:
        plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('Figure1_Preprocessing_Steps.png', dpi=300, bbox_inches='tight')
plt.close() 