import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data
# Adjust path as necessary (e.g., 'benchmark_results.csv')
df = pd.read_csv('../data/benchmark_4cores_new.csv')

# 2. Get the unique images (Total of 5)
images = df['image'].unique()

# 3. Initialize a large figure with subplots (3 rows, 2 columns)
# Increased height to accommodate a larger legend at the bottom
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
axes = axes.flatten()

# To store handles and labels for the global legend
handles, labels = [], []

for i, img in enumerate(images):
    ax = axes[i]
    img_data = df[df['image'] == img]
    
    # --- Sequential (Baseline) ---
    seq_data = img_data[img_data['solution'] == 'sequential']
    if not seq_data.empty:
        line, = ax.plot(seq_data['threads'], seq_data['speedup'], 'o-', label='Sequential', linewidth=2.5, color='black', zorder=10)
        if i == 0 and 'Sequential' not in labels: handles.append(line); labels.append('Sequential')
    
    # --- Basic Parallel ---
    basic_data = img_data[img_data['solution'] == 'basic_parallel'].sort_values('threads')
    if not basic_data.empty:
        line, = ax.plot(basic_data['threads'], basic_data['speedup'], 'o-', label='Basic Parallel', alpha=0.6)
        if i == 0 and 'Basic Parallel' not in labels: handles.append(line); labels.append('Basic Parallel')
        
    # --- Greedy Parallel (Multiple lines for batch sizes / seams removed) ---
    greedy_all = img_data[img_data['solution'] == 'greedy_parallel']
    # Filter for valid batch sizes (> 0) and sort them
    unique_batches = sorted([b for b in greedy_all['greedy_batch_size'].unique() if b > 0])
    
    for b in unique_batches:
        batch_data = greedy_all[greedy_all['greedy_batch_size'] == b].sort_values('threads')
        if not batch_data.empty:
            # We use a distinct marker '+' and a different color/alpha for these lines
            line, = ax.plot(batch_data['threads'], batch_data['speedup'], '+-', label=f'Greedy (batch {b})', alpha=0.8)
            if i == 0:
                lab = f'Greedy (batch {b})'
                if lab not in labels: handles.append(line); labels.append(lab)
            
    # --- Triangular One Way (Original Triangle) ---
    tri_one_data = img_data[img_data['solution'] == 'triangular_one_way_parallel']
    unique_tiles_one = sorted([t for t in tri_one_data['height_tiles'].unique() if 1 <= t <= 32])
    for tile in unique_tiles_one:
        tile_data = tri_one_data[tri_one_data['height_tiles'] == tile].sort_values('threads')
        line, = ax.plot(tile_data['threads'], tile_data['speedup'], 'x--', label=f'Tri One-Way ({tile} tiles)', alpha=0.7)
        if i == 0:
            lab = f'Tri One-Way ({tile} tiles)'
            if lab not in labels: handles.append(line); labels.append(lab)

    # --- Triangular Up-Down (Top-Bottom Parallel) ---
    tri_tb_data = img_data[img_data['solution'] == 'top_bottom_parallel']
    unique_tiles_tb = sorted([t for t in tri_tb_data['height_tiles'].unique() if 1 <= t <= 32])
    for tile in unique_tiles_tb:
        tile_data = tri_tb_data[tri_tb_data['height_tiles'] == tile].sort_values('threads')
        line, = ax.plot(tile_data['threads'], tile_data['speedup'], 'd:', label=f'Tri Up-Down ({tile} tiles)', alpha=0.7)
        if i == 0:
            lab = f'Tri Up-Down ({tile} tiles)'
            if lab not in labels: handles.append(line); labels.append(lab)
            
    # --- Formatting each subplot ---
    ax.set_title(f'Resolution: {img}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Threads (log scale)', fontsize=10)
    ax.set_ylabel('Speedup (log scale)', fontsize=10)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    thread_ticks = sorted(img_data['threads'].unique())
    ax.set_xticks(thread_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) 
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter()) 
    
    ax.grid(True, which="both", linestyle='--', alpha=0.4)

# 4. Handle the 6th subplot - Use it for the Legend
legend_ax = axes[5]
legend_ax.axis('off')
legend_ax.legend(handles, labels, loc='center', fontsize='large', frameon=True, title="All Solutions", ncol=2)

# 5. Finalize layout and save
plt.tight_layout()
plt.savefig('performance_comprehensive_greedy.png', dpi=300)