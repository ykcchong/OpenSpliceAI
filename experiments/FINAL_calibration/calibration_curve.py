import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict

def adjust_color_brightness(color, amount=0.5):
    import colorsys
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    rgb = mcolors.to_rgb(c)
    hls = colorsys.rgb_to_hls(*rgb)
    new_hls = (hls[0], max(0, min(1, amount * hls[1])), hls[2])
    new_rgb = colorsys.hls_to_rgb(*new_hls)
    return new_rgb

flanking_sizes = [80, 400, 2000, 10000]
classes = ["Non-splice site", "Acceptor site", "Donor site"]
base_colors = {80: 'blue', 400: 'green', 2000: 'orange', 10000: 'red'}

for species in ["MANE", "mouse", "honeybee", "arabidopsis", "zebrafish"]:
    base_dir = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/calibrate_outdir'
    output_dir = os.path.join(base_dir, f'{species}/viz/')
    os.makedirs(output_dir, exist_ok=True)

    for flanking_size in flanking_sizes:
        fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharex=True, sharey=True)
        
        base_color = base_colors.get(flanking_size, 'black')
        lighter_color = adjust_color_brightness(base_color, amount=1.5)
        darker_color = adjust_color_brightness(base_color, amount=0.8)

        temperature_file = os.path.join(base_dir, f'{species}/flanking_{flanking_size}/temperature.txt')
        try:
            with open(temperature_file, 'r') as f:
                temp_val = float(f.read().strip())
                temp_str = f"{temp_val:.4f}"
        except (FileNotFoundError, ValueError):
            temp_str = "N/A"

        for i, class_name in enumerate(classes):
            ax = axs[i]
            calibration_data_dir = os.path.join(
                base_dir, f'{species}/flanking_{flanking_size}/calibration_data'
            )
            
            # Load original
            orig_file = os.path.join(
                calibration_data_dir,
                f'calibration_data_{class_name}_original_{flanking_size}nt.npz'
            )
            data_orig = np.load(orig_file)
            prob_true_orig = data_orig['prob_true']
            prob_pred_orig = data_orig['prob_pred']

            # Load calibrated
            calib_file = os.path.join(
                calibration_data_dir,
                f'calibration_data_{class_name}_calibrated_{flanking_size}nt.npz'
            )
            data_calib = np.load(calib_file)
            prob_true_calib = data_calib['prob_true']
            prob_pred_calib = data_calib['prob_pred']

            # Plot original curve
            ax.plot(prob_pred_orig, prob_true_orig,
                    marker='o', linestyle='--', color=lighter_color,
                    label='Original (T=1.0000)')

            # Conditionally fill confidence interval if the keys exist
            if 'prob_true_lower' in data_orig and 'prob_true_upper' in data_orig:
                ax.fill_between(prob_pred_orig,
                                data_orig['prob_true_lower'],
                                data_orig['prob_true_upper'],
                                color=lighter_color, alpha=0.3,
                                label='CI (Original)')

            # Plot calibrated curve
            ax.plot(prob_pred_calib, prob_true_calib,
                    marker='o', linestyle='-', color=darker_color,
                    label=f'Calibrated (T={temp_str})')

            # Conditionally fill confidence interval if the keys exist
            if 'prob_true_lower' in data_calib and 'prob_true_upper' in data_calib:
                ax.fill_between(prob_pred_calib,
                                data_calib['prob_true_lower'],
                                data_calib['prob_true_upper'],
                                color=darker_color, alpha=0.3,
                                label='CI (Calibrated)')

            # Perfect calibration line
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_title(class_name, fontsize=15)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal', adjustable='box')
            if i == 0:
                ax.set_ylabel('Empirical Probability', fontsize=12)
            ax.set_xlabel('Predicted Probability', fontsize=12)
            ax.grid(True)

        # Combine legend
        handles, labels = axs[0].get_legend_handles_labels()
        for j in [1, 2]:
            h, l = axs[j].get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        legend_dict = OrderedDict()
        for h, l in zip(handles, labels):
            if l not in legend_dict:
                legend_dict[l] = h
        
        fig.legend(legend_dict.values(), legend_dict.keys(),
                   bbox_to_anchor=(0.5, 0.02),
                   loc='lower center',
                   ncol=4,
                   fontsize=11)
        
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        outname = f'calibration_curves_flanking_{flanking_size}nt_{species}.png'
        plt.savefig(os.path.join(output_dir, outname), dpi=300)
        plt.close()
