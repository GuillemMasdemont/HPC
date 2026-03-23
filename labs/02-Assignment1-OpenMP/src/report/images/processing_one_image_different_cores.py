import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data from all core configurations
core_files = {
    1: '../data/benchmark_1cores.csv',
    2: '../data/benchmark_2cores.csv',
    4: '../data/benchmark_4cores.csv',
    8: '../data/benchmark_8cores.csv'
}

dfs = []
for n_cores, filename in core_files.items():
    temp_df = pd.read_csv(filename)
    temp_df['hardware_cores'] = n_cores
    dfs.append(temp_df)

# Combine all data into a single DataFrame
df_all = pd.concat(dfs, ignore_index=True)

# 2. Select a single image for comparison (e.g., 3840x2160)
selected_image = '1920x1200.png'
image_df = df_all[df_all['image'] == selected_image]

# 3. Create a layout with subplots for each major parallel solution
# We'll plot Basic, Greedy, and Triangular (with 8 and 16 tiles)
solutions_to_compare = [
    ('basic_parallel', 0), 
    ('greedy_parallel', 0), 
    ('triangular_parallel', 8), 
    ('triangular_parallel', 16)
]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()

for i, (sol, tile_count) in enumerate(solutions_to_compare):
    ax = axes[i]
    sol_df = image_df[image_df['solution'] == sol]
    
    # Filter for specific tile count if it's the triangular solution
    if sol == 'triangular_parallel':
        sol_df = sol_df[sol_df['height_tiles'] == tile_count]
    
    # Plot a line for each hardware configuration (1, 2, 4, 8 cores)
    for hw_cores in sorted(sol_df['hardware_cores'].unique()):
        core_data = sol_df[sol_df['hardware_cores'] == hw_cores].sort_values('threads')
        label = f'{hw_cores} Cores'
        ax.plot(core_data['threads'], core_data['speedup'], 'o-', label=label, linewidth=2)

    # Formatting
    title_label = sol.replace('_', ' ').title()
    if tile_count > 0:
        title_label += f' ({tile_count} tiles)'
    
    ax.set_title(title_label, fontsize=14, fontweight='bold')
    ax.set_xlabel('Software Threads (Log Scale)', fontsize=11)
    ax.set_ylabel('Speedup Time (Log Scale)', fontsize=11)
    
    # Logarithmic scaling for both axes to see scaling efficiency
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Ensure ticks match thread counts (1, 2, 4, 8, 16, 32)
    thread_ticks = sorted(image_df['threads'].unique())
    ax.set_xticks(thread_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend(title="Hardware")

#plt.suptitle(f'Performance Comparison: {selected_image} across Hardware Cores', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'comparison_by_cores_{selected_image}', dpi=300)
#plt.show()