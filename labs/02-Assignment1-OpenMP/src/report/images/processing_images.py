import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the benchmark data
df = pd.read_csv('../data/benchmark_results.csv')

# 2. Get the unique images (Total of 5)
images = df['image'].unique()

# 3. Create plots for each image with log scale
for img in images:
    plt.clf()  # Clear current figure to avoid overlapping
    img_data = df[df['image'] == img]
    
    # Plot Sequential
    seq_data = img_data[img_data['solution'] == 'sequential']
    if not seq_data.empty:
        plt.plot(seq_data['threads'], seq_data['speedup'], 'o-', label='Sequential', linewidth=2)
    
    # Plot Basic Parallel
    basic_data = img_data[img_data['solution'] == 'basic_parallel'].sort_values('threads')
    if not basic_data.empty:
        plt.plot(basic_data['threads'], basic_data['speedup'], 'o-', label='Basic Parallel')
        
    # Plot Greedy Parallel
    greedy_data = img_data[img_data['solution'] == 'greedy_parallel'].sort_values('threads')
    if not greedy_data.empty:
        plt.plot(greedy_data['threads'], greedy_data['speedup'], 'o-', label='Greedy Parallel')
        
    # Plot Triangular Parallel for all tile configurations (4, 6, 8, 16)
    tri_data = img_data[img_data['solution'] == 'triangular_parallel']
    unique_tiles = sorted(tri_data['height_tiles'].unique())
    for tile in unique_tiles:
        tile_data = tri_data[tri_data['height_tiles'] == tile].sort_values('threads')
        plt.plot(tile_data['threads'], tile_data['speedup'], 'x--', label=f'Triangular ({tile} tiles)')
        
    # Formatting the plot
    plt.title(f'Performance Comparison: {img} (Log Scale)')
    plt.xlabel('Number of Threads (Logarithmic Scale)')
    plt.ylabel('Speedup')
    
    # Apply Log Scale (Base 2) to the X-axis
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)


    # Set explicit ticks to match the thread counts in the dataset
    thread_ticks = sorted(img_data['threads'].unique())
    plt.xticks(thread_ticks, labels=[str(t) for t in thread_ticks])
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    save_filename = f'log_speedup_{img.replace(".", "_")}'
    plt.savefig(save_filename)