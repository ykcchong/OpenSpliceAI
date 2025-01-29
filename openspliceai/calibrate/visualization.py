# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import os
from openspliceai.calibrate.calibrate_utils import *


def plot_score_distribution(probs, probs_scaled, labels, output_dir, index):
    index_names = {
        0: ('non-splice site', 'neither'),
        1: ('acceptor site', 'acceptor'),
        2: ('donor site', 'donor')
    }
    
    positive_indices = (labels == index)
    probs_positive = probs[:, index][positive_indices]
    probs_scaled_positive = probs_scaled[:, index][positive_indices]

    plt.figure(figsize=(5.5, 4))
    plt.hist(probs_positive, bins=30, alpha=0.5, label='Original')
    plt.hist(probs_scaled_positive, bins=30, alpha=0.5, label='Calibrated')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.title(f'Score Distribution for {index_names[index][0]}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration/prob_dist_{index_names[index][1]}.png")
    plt.close()

def plot_calibration_curves(calibration_data, calibration_data_scaled, classes, output_dir):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    for i, (class_name, (orig_data, scaled_data)) in enumerate(zip(classes, zip(calibration_data, calibration_data_scaled))):
        prob_true, prob_pred, bin_counts = orig_data
        prob_true_s, prob_pred_s, _ = scaled_data
        
        ci_lower, ci_upper = compute_confidence_intervals(prob_true, bin_counts)
        ci_lower_s, ci_upper_s = compute_confidence_intervals(prob_true_s, bin_counts)
        
        axs[i].plot(prob_pred, prob_true, 'o-', color='blue', label='Original')
        axs[i].fill_between(prob_pred, ci_lower, ci_upper, color='blue', alpha=0.1)
        axs[i].plot(prob_pred_s, prob_true_s, 'o-', color='green', label='Calibrated')
        axs[i].fill_between(prob_pred_s, ci_lower_s, ci_upper_s, color='green', alpha=0.1)
        axs[i].plot([0, 1], [0, 1], 'k--')
        axs[i].set_title(class_name)
        axs[i].set_xlabel('Predicted Probability')
        axs[i].set_ylabel('Empirical Probability')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration/calibration_curve.png", dpi=300)
    plt.close()

def plot_brier_scores(brier_uncal, brier_cal, classes, output_dir):
    plt.figure(figsize=(8, 6))
    x = np.arange(len(classes))
    plt.bar(x - 0.2, brier_uncal, 0.4, label='Uncalibrated')
    plt.bar(x + 0.2, brier_cal, 0.4, label='Calibrated')
    plt.xticks(x, classes)
    plt.ylabel('Brier Score')
    plt.title('Brier Score Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration/brier_scores.png", dpi=300)
    plt.close()

def plot_calibration_map(scaled_model, device, output_dir):
    p1d = np.linspace(0, 1, 20)
    p0, p1 = np.meshgrid(p1d, p1d)
    p2 = 1 - p0 - p1
    mask = (p2 >= 0) & (p2 <= 1)
    p = np.vstack([p0[mask], p1[mask], p2[mask]]).T
    
    logits = reverse_softmax(p)
    with torch.no_grad():
        logits_scaled = scaled_model.temperature_scale(torch.tensor(logits).float().to(device))
    probs_scaled = torch.nn.functional.softmax(logits_scaled, dim=1).cpu().numpy()

    plt.figure(figsize=(4.8, 4.8))
    colors = ['red', 'green', 'blue']
    for i in range(p.shape[0]):
        color = colors[np.argmax(p[i])]
        plt.arrow(p[i][1], p[i][2], 
                 probs_scaled[i][1]-p[i][1], probs_scaled[i][2]-p[i][2],
                 color=color, head_width=0.01, length_includes_head=True)
    
    plt.plot([0, 1], [0, 0], 'k-')
    plt.plot([1, 0], [0, 1], 'k-')
    plt.plot([0, 0], [1, 0], 'k-')
    plt.xlabel("Acceptor Probability")
    plt.ylabel("Donor Probability")
    plt.title("Calibration Map")
    plt.legend(handles=[
        plt.Line2D([0], [0], color='red', label='Non-splice'),
        plt.Line2D([0], [0], color='green', label='Acceptor'),
        plt.Line2D([0], [0], color='blue', label='Donor')
    ])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration/calibration_map.png", dpi=300)
    plt.close()