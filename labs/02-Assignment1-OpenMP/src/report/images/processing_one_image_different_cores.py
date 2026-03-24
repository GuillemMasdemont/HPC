import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data from all core configurations
# Note: Ensure these files exist in your ../data/ directory
core_files = {
    4: '../data/benchmark_4cores_new.csv',
    8: '../data/benchmark_8cores_new.csv',
    16: '../data/benchmark_16threads copy.csv',
    32: '../data/benchmark_32cores_new.csv'
}

dfs = []
for n_cores, filename in core_files.items():
    try:
        temp_df = pd.read_csv(filename)
        temp_df['hardware_cores'] = n_cores
        dfs.append(temp_df)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Skipping {n_cores} core data.")

# Combine all data into a single DataFrame
if not dfs:
    print("No data loaded. Check file paths.")
    exit()
    
df_all = pd.concat(dfs, ignore_index=True)

# 2. Select a single image for comparison
selected_image = '1920x1200.png'
image_df = df_all[df_all['image'] == selected_image]

# 3. Create a layout with subplots for each major parallel solution
# Mapping based on your CSV labels:
# 'triangular_parallel' -> 'triangular_one_way_parallel'
solutions_to_compare = [
    ('basic_parallel', 0), 
    ('greedy_parallel', 16),               # Compare Greedy with batch size 16
    ('triangular_one_way_parallel', 8),    # Triangular with 8 tiles
    ('triangular_one_way_parallel', 16)    # Triangular with 16 tiles
]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()

for i, (sol, param) in enumerate(solutions_to_compare):
    ax = axes[i]
    sol_df = image_df[image_df['solution'] == sol]
    
    # Filter for specific tile count or batch size to avoid overlapping lines
    if 'triangular' in sol:
        sol_df = sol_df[sol_df['height_tiles'] == param]
        title_suffix = f' ({param} tiles)'
    elif sol == 'greedy_parallel':
        sol_df = sol_df[sol_df['greedy_batch_size'] == param]
        title_suffix = f' (batch {param})'
    else:
        title_suffix = ""

    # Plot a line for each hardware configuration
    hw_configs = sorted(sol_df['hardware_cores'].unique())
    for hw_cores in hw_configs:
        core_data = sol_df[sol_df['hardware_cores'] == hw_cores].sort_values('threads')
        if not core_data.empty:
            label = f'{hw_cores} Cores'
            ax.plot(core_data['threads'], core_data['speedup'], 'o-', label=label, linewidth=2)

    # Formatting
    title_label = sol.replace('_', ' ').title() + title_suffix
    ax.set_title(title_label, fontsize=17, fontweight='bold')
    ax.set_xlabel('Software Threads', fontsize=13)
    ax.set_ylabel('Speedup Factor', fontsize=13)
    
    # Logarithmic scaling (Base 2) for both axes
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Set ticks to match the thread counts (1, 2, 4, 8, 16, 32...)
    thread_ticks = sorted(image_df['threads'].unique())
    ax.set_xticks(thread_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend(title="Hardware", fontsize='large')

# Add a global title to the figure
#plt.suptitle(f'Hardware Core Comparison: {selected_image}\n(Scaling Efficiency across Software Threads)', fontsize=18, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig(f'comparison_by_cores_{selected_image}.png', dpi=300)
# plt.show()