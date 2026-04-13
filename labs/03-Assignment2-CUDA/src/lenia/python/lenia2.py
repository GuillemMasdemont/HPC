import matplotlib.pyplot as plt
import numpy as np

# Extracting the Speedup(v3) data from the tables
N = np.array([128, 256, 512, 1024, 2048, 4096])

speedup_8 = np.array([1643.31, 2591.90, 2952.69, 3688.49, 3969.52, 4002.36])
speedup_16 = np.array([1886.26, 3876.76, 4778.02, 6926.76, 7607.37, 7692.49])
speedup_32 = np.array([1907.48, 3821.09, 4711.18, 6537.52, 7375.32, 7464.71])

# Global font settings for report readability
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

plt.figure(figsize=(12, 7))

x = np.arange(len(N))
width = 0.25 # Width of the bars

# Create grouped bars
plt.bar(x - width, speedup_8, width, label='Tile Size 8', color='#1f77b4', edgecolor='black')
plt.bar(x, speedup_16, width, label='Tile Size 16', color='#2ca02c', edgecolor='black')
plt.bar(x + width, speedup_32, width, label='Tile Size 32', color='#ff7f0e', edgecolor='black')

# Formatting and labels
plt.xlabel('Grid Size (N x N)')
plt.ylabel('Speedup Factor (V3 vs CPU)')
plt.title('Speedup Performance Comparison Across Tile Sizes')
plt.xticks(x, [str(n) for n in N])

# Place legend in upper left to avoid covering the bars
plt.legend(title='Block Dimension', loc='upper left', frameon=True, edgecolor='black')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('tile_speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.show()