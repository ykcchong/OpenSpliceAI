import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# species = ['arabidopsis', 'mouse', 'zebrafish', 'honeybee', 'MANE',]
species = ['arabidopsis']
species_to_name_map = {
    'arabidopsis': 'Arabidopsis',
    'mouse': 'Mouse',
    'zebrafish': 'Zebrafish',
    'honeybee': 'Honeybee',
    'MANE': 'Human',
}

base_dir = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/calibrate_outdir"
os.makedirs(f'{base_dir}/viz/', exist_ok=True)
for s in species:
    root_dir = f"{base_dir}/{s}"

    # Get a sorted list of flanking directories (assuming names like "flanking_10000", etc.)
    flanking_dirs = sorted(
        glob.glob(f'{root_dir}/flanking_*'),
        key=lambda d: int(os.path.basename(d).split('_')[1])
    )

    n_dirs = len(flanking_dirs)
    if n_dirs == 0:
        raise ValueError("No flanking directories found.")

    # Create a figure with n_dirs rows and 2 columns.
    # Note: the left column (calibration curve) is wider than the right (scaling map).
    fig, axs = plt.subplots(n_dirs, 2, 
                            figsize=(20, 6 * n_dirs),
                            gridspec_kw={'width_ratios': [20, 4.8]})

    # Add a full-plot 
    if s == 'arabidopsis':
        fig.suptitle(r'Calibration results for $\mathit{Arabidopsis}$', fontsize=28)
    else:
        fig.suptitle(f"Calibration results for {species_to_name_map[s]}", fontsize=28)

    # If there is only one row, make sure axs is indexable as 2D.
    if n_dirs == 1:
        axs = [axs]

    # A counter to label subplots from A, B, ... etc.
    letter_counter = 0

    for i, d in enumerate(flanking_dirs):
        # Construct paths for the images in the "calibration" subfolder.
        curve_path = os.path.join(d, 'calibration', 'calibration_curve.png')
        map_path = os.path.join(d, 'calibration', 'temperature_scaling_calibration_map.png')
        
        # Check that the files exist.
        if not os.path.exists(curve_path):
            raise FileNotFoundError(f"File not found: {curve_path}")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"File not found: {map_path}")
        
        # Read images.
        curve_img = mpimg.imread(curve_path)
        map_img = mpimg.imread(map_path)
        
        # Plot the calibration curve on the left.
        ax_curve = axs[i][0] if n_dirs > 1 else axs[0]
        ax_curve.imshow(curve_img)
        ax_curve.axis('off')
        # ax_curve.set_title(f'Calibration Curve ({os.path.basename(d)})', fontsize=16)
        flanking_size = os.path.basename(d).split('_')[1]
        # I want shift the title a bit to the right
        ax_curve.set_title(f'Calibration Curve (Flanking sequence size {flanking_size})', fontsize=20, x=0.5)
        # Annotate with a letter (upper left corner)
        letter = chr(ord('A') + letter_counter)
        letter_counter += 1
        ax_curve.text(-0.02, 1.02, letter, transform=ax_curve.transAxes,
                    fontsize=30, fontweight='bold', va='top', ha='left')
        
        # Plot the temperature scaling calibration map on the right.
        ax_map = axs[i][1] if n_dirs > 1 else axs[1]
        ax_map.imshow(map_img)
        ax_map.axis('off')
        # ax_map.set_title(f'Temperature Scaling Map ({os.path.basename(d)})', fontsize=16)
        # Annotate with a letter (upper left corner)
        letter = chr(ord('A') + letter_counter)
        letter_counter += 1
        ax_map.text(-0.1, 1.1, letter, transform=ax_map.transAxes,
                    fontsize=30, fontweight='bold', va='top', ha='left')

    # Adjust layout to add extra horizontal space between columns.
    plt.tight_layout(w_pad=3.0)
    plt.savefig(f'{base_dir}/viz/{s}_calibration_plots.png')
    plt.show()
