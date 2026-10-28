# preliminary testing data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Load the CSV file
df = pd.read_csv('preliminary_test_relative_velocity.csv')

# 2. Define a mask for missing/zero data (handling both NaN and 0 values)
is_invalid = (
    df['robot_x_px'].isna() | (df['robot_x_px'] == 0) |
    df['robot_y_px'].isna() | (df['robot_y_px'] == 0)
)

# A row is valid when coordinates are actively recorded
valid_mask = ~is_invalid

# 3. Create unique trial IDs
# Each time we transition from an invalid/zero region into a valid trial region, 
# we increment our trial counter.
trial_id = (valid_mask & ~valid_mask.shift(1).fillna(False)).cumsum()

# 4. Split into a list of DataFrames
trials = []
# Group by the trial ID, keeping only the valid rows for each trial
for tid, group in df[valid_mask].groupby(trial_id[valid_mask]):
    # Optional: Filter out any accidental tiny blips if they are below your 50-row threshold
    if len(group) > 75:  # Adjust minimum trial length as needed
        trials.append(group.reset_index(drop=True))

print(f"Successfully extracted {len(trials)} trials into the list.")

# Graphing
output_dir = 'preliminary_test_charts'
os.makedirs(output_dir, exist_ok=True)

# List of all 16 columns to generate figures for
columns_to_plot = [
    'human_x_px', 'human_y_px', 'robot_x_px', 'robot_y_px', 
    'human_vx_px_s', 'human_vy_px_s', 'robot_vx_px_s', 'robot_vy_px_s', 
    'relative_x_px', 'relative_y_px', 'distance_px', 
    'relative_vx_px_s', 'relative_vy_px_s', 'relative_speed_px_s', 
    'separation_rate_px_s', 'closing_speed_px_s'
]

target_points = ['nose', 'torso']
marker_styles = {
    'nose': 'o',   # Circle for nose
    'torso': '^'   # Triangle for torso
}

# Loop through each column to create a separate figure
for col in columns_to_plot:
    # Create a 3x3 grid of subplots for the 9 trials
    fig, axes = plt.subplots(2, 5, figsize=(15, 12), sharex=False, sharey=True)
    axes_flat = axes.flatten()
    
    # Iterate through each trial and assign it to a grid cell
    for trial_idx, trial_df in enumerate(trials):
        ax = axes_flat[trial_idx]
        
        # Filter out wrists, keeping only nose and torso
        filtered_df = trial_df[trial_df['point_name'].isin(target_points)]
        
        # Group by point_name to plot with distinct markers
        for point_name, group in filtered_df.groupby('point_name'):
            ax.scatter(
                group['frame_number'],
                group[col],
                marker=marker_styles.get(point_name, 'o'),
                label=point_name,
                alpha=0.7,
                s=15
            )
            
        # Subplot formatting
        ax.set_title(f'Trial {trial_idx + 1}', fontsize=10)
        ax.set_xlabel('Frame Number', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend and y ax label only to the first subplot to prevent cluttering the grid
        if trial_idx == 0:
            ax.set_ylabel(col, fontsize=9)
            ax.legend(fontsize='small', loc='best')
            
    # Overall figure layout adjustments
    fig.suptitle(f'{col}', fontsize=16, y=0.93)
    file_path = os.path.join(output_dir, f'{col}_trials_grid.png')
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

print(f"Successfully generated and saved all charts to the '{output_dir}' folder!")