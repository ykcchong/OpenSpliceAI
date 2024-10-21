import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def adjust_color_brightness(color, amount=0.5):
    """
    Adjust the brightness of a given color.
    amount > 1 increases brightness, amount < 1 decreases brightness.
    """
    import colorsys
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    rgb = mcolors.to_rgb(c)
    hls = colorsys.rgb_to_hls(*rgb)
    # Adjust luminance
    new_hls = (hls[0], max(0, min(1, amount * hls[1])), hls[2])
    new_rgb = colorsys.hls_to_rgb(*new_hls)
    return new_rgb

# List of flanking sizes and species to include
flanking_sizes = [80, 400, 2000, 10000]
classes = ["Non-splice site", "Acceptor site", "Donor site"]
for species in ["MANE", "mouse", "honeybee", "arabidopsis", "zebrafish"]:
# species = "mouse"  # Replace with your species name

    # Base directory where calibration data is stored
    base_dir = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/calibrate_outdir'
    output_dir = os.path.join(base_dir, f'{species}/viz/')
    os.makedirs(output_dir, exist_ok=True)

    # Define base colors for each flanking size
    base_colors = {
        80: 'blue',
        400: 'green',
        2000: 'orange',
        10000: 'red'
    }

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 7), sharex=True, sharey=True)

    # Loop over classes
    for i, class_name in enumerate(classes):
        ax = axs[i]
        for flanking_size in flanking_sizes:
            calibration_data_dir = os.path.join(base_dir, f'{species}/flanking_{flanking_size}/calibration_data')
            
            # Path to the temperature file
            temperature_file = os.path.join(base_dir, f'{species}/flanking_{flanking_size}/temperature.txt')
            
            # Read the temperature value
            try:
                with open(temperature_file, 'r') as f:
                    temperature_value = f.read().strip()
                    temperature_value_formatted = f"{float(temperature_value):.4f}"
            except (FileNotFoundError, ValueError):
                print(f"Temperature file not found or invalid for flanking size {flanking_size}.")
                temperature_value_formatted = 'N/A'
            
            # Load original calibration data
            original_data_file = os.path.join(calibration_data_dir, f'calibration_data_{class_name}_original_{flanking_size}nt.npz')
            data_original = np.load(original_data_file)
            prob_true_original = data_original['prob_true']
            prob_pred_original = data_original['prob_pred']

            # Load calibrated calibration data
            calibrated_data_file = os.path.join(calibration_data_dir, f'calibration_data_{class_name}_calibrated_{flanking_size}nt.npz')
            data_calibrated = np.load(calibrated_data_file)
            prob_true_calibrated = data_calibrated['prob_true']
            prob_pred_calibrated = data_calibrated['prob_pred']

            # Get base color for this flanking size
            base_color = base_colors.get(flanking_size, 'black')  # Default to black if not found

            # Adjust colors
            lighter_color = adjust_color_brightness(base_color, amount=1.5)  # Lighter color for original
            darker_color = adjust_color_brightness(base_color, amount=0.8)   # Darker color for calibrated

            # Plot original calibration curve (dashed line, lighter color)
            ax.plot(prob_pred_original, prob_true_original, marker='o', linestyle='--',
                    color=lighter_color, label=f'Original     {flanking_size}nt (T=1.0000)')

            # Plot calibrated calibration curve (solid line, darker color)
            ax.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', linestyle='-',
                    color=darker_color, label=f'Calibrated {flanking_size}nt (T={temperature_value_formatted})')

        # Plot the diagonal line for perfect calibration
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line

        # Set axis limits to be equal
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Ensure the aspect ratio is equal
        ax.set_aspect('equal', adjustable='box')

        # Set plot titles and labels
        ax.set_title(f'{class_name}', fontsize=22, fontweight=400)
        if i == 0:
            ax.set_ylabel('Empirical Probability', fontsize=16)
        ax.set_xlabel('Predicted Probability', fontsize=16)

        ax.grid(True)

    # Create a combined legend
    handles, labels = axs[0].get_legend_handles_labels()
    # Remove duplicate labels
    from collections import OrderedDict
    legend_dict = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in legend_dict:
            legend_dict[label] = handle

    # Place the legend outside the plot area
    fig.legend(legend_dict.values(), legend_dict.keys(), bbox_to_anchor=(0.5, -0.01),loc='lower center', ncol=4, fontsize=12)

    # Adjust the layout to accommodate the legend
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.00, right=1, hspace=0.3, wspace=0.3)
    # # Adjust layout and legend
    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Adjust rect to leave space for the legend
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'calibration_curve_comparison_subplots.png'), dpi=300)
    plt.close()
