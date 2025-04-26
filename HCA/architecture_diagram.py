import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def create_architecture_diagram():
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 15)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#FFB6C1',  # Light pink
        'conv': '#87CEEB',   # Sky blue
        'pool': '#98FB98',   # Pale green
        'dense': '#FFA07A',  # Light salmon
        'output': '#DDA0DD', # Plum
        'arrow': '#000000',  # Black
        'text': '#000000'    # Black
    }

    # Define vertical positions
    y_positions = {
        'title': 14.5,
        'input': 13.0,
        'conv1': 11.5,
        'conv2': 10.0,
        'conv3': 8.5,
        'conv4': 7.0,
        'dense': 5.5,
        'output': 4.0,
        'details': 2.5
    }

    # Draw main title
    plt.title('Multi-Block CNN Architecture for ECG Classification', fontsize=14, pad=20)

    # Draw Input Layer
    input_box = patches.Rectangle((1, y_positions['input']), 2, 0.8, 
                                 linewidth=1, edgecolor='black', facecolor=colors['input'])
    ax.add_patch(input_box)
    ax.text(2, y_positions['input'] + 0.4, 'Input Layer\n(187, 1)', 
            ha='center', va='center', fontsize=10)

    # Draw Convolutional Blocks
    conv_blocks = [
        {'x': 1, 'y': y_positions['conv1'], 'text': 'Conv1\n64 filters, kernel=5\nBatchNorm\nDropout(0.3)', 'color': colors['conv']},
        {'x': 1, 'y': y_positions['conv2'], 'text': 'Conv2\n128 filters, kernel=3\nBatchNorm\nDropout(0.3)', 'color': colors['conv']},
        {'x': 1, 'y': y_positions['conv3'], 'text': 'Conv3\n256 filters, kernel=3\nBatchNorm\nDropout(0.3)', 'color': colors['conv']},
        {'x': 1, 'y': y_positions['conv4'], 'text': 'Conv4\n512 filters, kernel=3\nBatchNorm\nDropout(0.3)', 'color': colors['conv']}
    ]

    for block in conv_blocks:
        rect = patches.Rectangle((block['x'], block['y']), 2, 0.8,
                                linewidth=1, edgecolor='black', facecolor=block['color'])
        ax.add_patch(rect)
        ax.text(block['x'] + 1, block['y'] + 0.4, block['text'],
                ha='center', va='center', fontsize=9)

    # Draw MaxPooling layers
    pool_positions = [
        {'x': 1, 'y': y_positions['conv1'] - 0.5, 'text': 'MaxPool1\npool_size=2'},
        {'x': 1, 'y': y_positions['conv2'] - 0.5, 'text': 'MaxPool2\npool_size=2'},
        {'x': 1, 'y': y_positions['conv3'] - 0.5, 'text': 'MaxPool3\npool_size=2'},
        {'x': 1, 'y': y_positions['conv4'] - 0.5, 'text': 'MaxPool4\npool_size=2'}
    ]

    for pool in pool_positions:
        rect = patches.Rectangle((pool['x'], pool['y']), 2, 0.4,
                                linewidth=1, edgecolor='black', facecolor=colors['pool'])
        ax.add_patch(rect)
        ax.text(pool['x'] + 1, pool['y'] + 0.2, pool['text'],
                ha='center', va='center', fontsize=8)

    # Draw Dense Layers
    dense_layers = [
        {'x': 1, 'y': y_positions['dense'], 'text': 'Dense1\n512 units\nDropout(0.5)', 'color': colors['dense']},
        {'x': 1, 'y': y_positions['dense'] - 0.5, 'text': 'Dense2\n256 units\nDropout(0.5)', 'color': colors['dense']}
    ]

    for layer in dense_layers:
        rect = patches.Rectangle((layer['x'], layer['y']), 2, 0.4,
                                linewidth=1, edgecolor='black', facecolor=layer['color'])
        ax.add_patch(rect)
        ax.text(layer['x'] + 1, layer['y'] + 0.2, layer['text'],
                ha='center', va='center', fontsize=8)

    # Draw Output Layer
    output_box = patches.Rectangle((1, y_positions['output']), 2, 0.8,
                                  linewidth=1, edgecolor='black', facecolor=colors['output'])
    ax.add_patch(output_box)
    ax.text(2, y_positions['output'] + 0.4, 'Output Layer\n5 classes\nSoftmax',
            ha='center', va='center', fontsize=10)

    # Draw arrows
    arrows = [
        # Input to Conv1
        {'start': (2, y_positions['input']), 'end': (2, y_positions['conv1'] + 0.8), 'text': '↓'},
        # Conv1 to Pool1
        {'start': (2, y_positions['conv1']), 'end': (2, y_positions['conv1'] - 0.1), 'text': '↓'},
        # Pool1 to Conv2
        {'start': (2, y_positions['conv1'] - 0.5), 'end': (2, y_positions['conv2'] + 0.8), 'text': '↓'},
        # Conv2 to Pool2
        {'start': (2, y_positions['conv2']), 'end': (2, y_positions['conv2'] - 0.1), 'text': '↓'},
        # Pool2 to Conv3
        {'start': (2, y_positions['conv2'] - 0.5), 'end': (2, y_positions['conv3'] + 0.8), 'text': '↓'},
        # Conv3 to Pool3
        {'start': (2, y_positions['conv3']), 'end': (2, y_positions['conv3'] - 0.1), 'text': '↓'},
        # Pool3 to Conv4
        {'start': (2, y_positions['conv3'] - 0.5), 'end': (2, y_positions['conv4'] + 0.8), 'text': '↓'},
        # Conv4 to Pool4
        {'start': (2, y_positions['conv4']), 'end': (2, y_positions['conv4'] - 0.1), 'text': '↓'},
        # Pool4 to Dense1
        {'start': (2, y_positions['conv4'] - 0.5), 'end': (2, y_positions['dense'] + 0.4), 'text': '↓'},
        # Dense1 to Dense2
        {'start': (2, y_positions['dense']), 'end': (2, y_positions['dense'] - 0.1), 'text': '↓'},
        # Dense2 to Output
        {'start': (2, y_positions['dense'] - 0.5), 'end': (2, y_positions['output'] + 0.8), 'text': '↓'}
    ]

    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], linewidth=1.5))
        ax.text((arrow['start'][0] + arrow['end'][0])/2,
                (arrow['start'][1] + arrow['end'][1])/2,
                arrow['text'], ha='center', va='center', fontsize=10)

    # Add model details
    details = [
        {'x': 4, 'y': 13.0, 'text': 'Model Configuration:', 'fontsize': 10, 'fontweight': 'bold'},
        {'x': 4, 'y': 12.0, 'text': 'Total Parameters: 2,847,749', 'fontsize': 9},
        {'x': 4, 'y': 11.5, 'text': 'Trainable Parameters: 2,845,701', 'fontsize': 9},
        {'x': 4, 'y': 11.0, 'text': 'Non-trainable Parameters: 2,048', 'fontsize': 9},
        {'x': 4, 'y': 10.0, 'text': 'Regularization:', 'fontsize': 10, 'fontweight': 'bold'},
        {'x': 4, 'y': 9.5, 'text': 'L2 Regularization (weight decay = 0.001)', 'fontsize': 9},
        {'x': 4, 'y': 9.0, 'text': 'Dropout: 0.3 (Conv), 0.5 (Dense)', 'fontsize': 9},
        {'x': 4, 'y': 8.5, 'text': 'Batch Normalization after each layer', 'fontsize': 9}
    ]

    for detail in details:
        ax.text(detail['x'], detail['y'], detail['text'],
                ha='left', va='center', fontsize=detail.get('fontsize', 9),
                fontweight=detail.get('fontweight', 'normal'))

    plt.savefig('figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_architecture_diagram() 