import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data
# Adjust path as necessary (e.g., 'benchmark_results.csv')
df = pd.read_csv('../data/benchmark_4cores.csv')


# 2. Get the unique images (Total of 5)
images = df['image'].unique()
n_images = len(images)

# 3. Initialize a large figure with subplots (3 rows, 2 columns)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

# To store handles and labels for the global legend
handles, labels = [], []

for i, img in enumerate(images):
    ax = axes[i]
    img_data = df[df['image'] == img]
    
    # --- Plotting the different solutions ---
    
    # Sequential (Baseline)
    seq_data = img_data[img_data['solution'] == 'sequential']
    if not seq_data.empty:
        line, = ax.plot(seq_data['threads'], seq_data['speedup'], 'o-', label='Sequential', linewidth=2, color='black')
        if i == 0: handles.append(line); labels.append('Sequential')
    
    # Basic Parallel
    basic_data = img_data[img_data['solution'] == 'basic_parallel'].sort_values('threads')
    if not basic_data.empty:
        line, = ax.plot(basic_data['threads'], basic_data['speedup'], 'o-', label='Basic Parallel')
        if i == 0: handles.append(line); labels.append('Basic Parallel')
        
    # Greedy Parallel
    greedy_data = img_data[img_data['solution'] == 'greedy_parallel'].sort_values('threads')
    if not greedy_data.empty:
        line, = ax.plot(greedy_data['threads'], greedy_data['speedup'], 'o-', label='Greedy Parallel')
        if i == 0: handles.append(line); labels.append('Greedy Parallel')
        
    # Triangular Parallel (for all tile counts)
    tri_data = img_data[img_data['solution'] == 'triangular_parallel']
    unique_tiles = sorted(tri_data['height_tiles'].unique())
    for tile in unique_tiles:
        tile_data = tri_data[tri_data['height_tiles'] == tile].sort_values('threads')
        line, = ax.plot(tile_data['threads'], tile_data['speedup'], 'x--', label=f'Triangular ({tile} tiles)')
        if i == 0: handles.append(line); labels.append(f'Triangular ({tile} tiles)')
        
    # --- Formatting each subplot ---
    ax.set_title(f'Resolution: {img}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Threads (Log Scale)', fontsize=10)
    ax.set_ylabel('Speedup Time (Log Scale)', fontsize=10)
    
    # Apply Log Scale (Base 2) to both axes
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # Set explicit ticks to match the thread counts in the dataset
    thread_ticks = sorted(img_data['threads'].unique())
    ax.set_xticks(thread_ticks)
    # Ensure ticks show as integers (1, 2, 4...) rather than scientific notation
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) 
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter()) 
    
    ax.grid(True, which="both", linestyle='--', alpha=0.4)

# 4. Handle the 6th subplot (index 5) - Use it for the Legend
legend_ax = axes[5]
legend_ax.axis('off') # Hide the axis lines and labels
legend_ax.legend(handles, labels, loc='center', fontsize='large', frameon=True, title="All Solutions")

# 5. Finalize layout and save
plt.tight_layout()
plt.savefig('merged_benchmark_plots.png', dpi=300)
#plt.show()