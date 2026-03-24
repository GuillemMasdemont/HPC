import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data
# Adjust path as necessary
try:
    df = pd.read_csv('../data/benchmark_32cores_new.csv')
except FileNotFoundError:
    print("File not found. Please ensure the path is correct.")
    # For demonstration, we'll assume df is loaded
    exit()

# 2. Get the first image only
images = df['image'].unique()
if len(images) == 0:
    print("No images found in the dataset.")
    exit()

img = images[0]  # Select the first image
img_data = df[df['image'] == img]

# 3. Initialize a single figure
plt.figure(figsize=(8, 6))
ax = plt.gca()

# To store handles and labels for the legend
handles, labels = [], []

# --- Sequential (Baseline) ---
seq_data = img_data[img_data['solution'] == 'sequential']
if not seq_data.empty:
    line, = ax.plot(seq_data['threads'], seq_data['speedup'], 'o-', 
                    label='Sequential', linewidth=2.5, color='black', zorder=10)
    handles.append(line); labels.append('Sequential')

# --- Basic Parallel ---
basic_data = img_data[img_data['solution'] == 'basic_parallel'].sort_values('threads')
if not basic_data.empty:
    line, = ax.plot(basic_data['threads'], basic_data['speedup'], 'o-', 
                    label='Basic Parallel', alpha=0.6)
    handles.append(line); labels.append('Basic Parallel')
    
# --- Greedy Parallel ---
greedy_all = img_data[img_data['solution'] == 'greedy_parallel']
unique_batches = sorted([b for b in greedy_all['greedy_batch_size'].unique() if b > 0])
for b in unique_batches:
    batch_data = greedy_all[greedy_all['greedy_batch_size'] == b].sort_values('threads')
    if not batch_data.empty:
        line, = ax.plot(batch_data['threads'], batch_data['speedup'], '+-', 
                        label=f'Greedy (batch {b})', alpha=0.8)
        handles.append(line); labels.append(f'Greedy (batch {b})')
        
# --- Triangular One Way ---
tri_one_data = img_data[img_data['solution'] == 'triangular_one_way_parallel']
unique_tiles_one = sorted([t for t in tri_one_data['height_tiles'].unique() if 1 <= t <= 32])
for tile in unique_tiles_one:
    tile_data = tri_one_data[tri_one_data['height_tiles'] == tile].sort_values('threads')
    line, = ax.plot(tile_data['threads'], tile_data['speedup'], 'x--', 
                    label=f'Tri One-Way ({tile} tiles)', alpha=0.7)
    handles.append(line); labels.append(f'Tri One-Way ({tile} tiles)')

# --- Triangular Up-Down ---
tri_tb_data = img_data[img_data['solution'] == 'top_bottom_parallel']
unique_tiles_tb = sorted([t for t in tri_tb_data['height_tiles'].unique() if 1 <= t <= 32])
for tile in unique_tiles_tb:
    tile_data = tri_tb_data[tri_tb_data['height_tiles'] == tile].sort_values('threads')
    line, = ax.plot(tile_data['threads'], tile_data['speedup'], 'd:', 
                    label=f'Tri Up-Down ({tile} tiles)', alpha=0.7)
    handles.append(line); labels.append(f'Tri Up-Down ({tile} tiles)')
        
# --- Formatting ---
ax.set_title(f'Performance Benchmark: {img}', fontsize=14, fontweight='bold')
ax.set_xlabel('Threads (log scale)', fontsize=12)
ax.set_ylabel('Speedup (log scale)', fontsize=12)

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)

thread_ticks = sorted(img_data['threads'].unique())
ax.set_xticks(thread_ticks)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) 
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter()) 

ax.grid(True, which="both", linestyle='--', alpha=0.4)
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')

plt.tight_layout()
plt.savefig('performance_first_image.png', dpi=300)
#plt.show()