import torch
import time
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from openspliceai.calibrate.openspliceai_calibrate import *
from openspliceai.calibrate.temperature_scaling import *
from openspliceai.train_base.utils import *
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve 
from sklearn.metrics import brier_score_loss

def compute_calibration_curve(labels, probs, n_bins=10, strategy='quantile'):
    # Ensure minimum samples per bin
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy=strategy)
    # Compute bin counts
    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(probs, quantiles)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1  # Adjust indices
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
    return prob_true, prob_pred, bin_counts


def compute_confidence_intervals(prob_true, bin_counts, z=1.96):
    ci_lower = []
    ci_upper = []
    for p, n in zip(prob_true, bin_counts):
        if n > 0:
            std_error = np.sqrt(p * (1 - p) / n)
            delta = z * std_error
        else:
            delta = 0
        ci_lower.append(max(p - delta, 0))
        ci_upper.append(min(p + delta, 1))
    return np.array(ci_lower), np.array(ci_upper)


def reverse_softmax(probs):
    # Convert probabilities back to logits
    # Avoid division by zero
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs)
    return logits


def score_frequency_distribution(probs, probs_scaled, labels, outdir, index=1):
    # Flatten the tensors for simplicity
    probabilities_flat = probs[:, index]
    probabilities_scaled_flat = probs_scaled[:, index]
    print("probabilities_flat.shape: ", probabilities_flat.shape)
    print("probabilities_scaled_flat.shape: ", probabilities_scaled_flat.shape)
    print("labels.shape: ", labels.shape)
    # Filter out cases where the label is zero (i.e., keep only positive cases for Class 1)
    positive_indices = (labels == index)  # True for positive cases
    print("positive_indices: ", positive_indices)
    print("len(positive_indices): ", len(positive_indices))
    probabilities_positive = probabilities_flat[positive_indices]
    probabilities_scaled_positive = probabilities_scaled_flat[positive_indices]
    labels_positive = labels[positive_indices]
    print("len(probabilities_positive): ", len(probabilities_positive))
    print("probabilities_positive: ", probabilities_positive)
    print("len(probabilities_scaled_positive): ", len(probabilities_scaled_positive))
    print("probabilities_scaled_positive: ", probabilities_scaled_positive)
    print("len(labels_positive): ", len(labels_positive))
    print("labels_positive: ", labels_positive)
    # Since we're now only looking at positive cases, the labels will be all ones,
    # and we're primarily interested in the distribution of predicted probabilities.
    # Plotting
    plt.figure(figsize=(5.5, 4))
    # Probability distribution for Class 1 from logits (only positive cases)
    plt.hist(probabilities_positive, bins=30, alpha=0.5, label='Predicted Probabilities')
    plt.hist(probabilities_scaled_positive, bins=30, alpha=0.5, label='Predicted Calibrated Probabilities')
    # For positive cases, since all labels are 1, we can simply indicate the average or density with a line or annotation
    # plt.axvline(x=np.mean(labels_positive), color='r', linestyle='dashed', linewidth=2, label='Actual Labels')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    if index == 0:
        plt.title('Score Distribution for non-splice site')
        plt.tight_layout()
        plt.savefig(f"{outdir}/calibration/prob_dist_neither.png")
    if index == 1:
        plt.title('Score Distribution for acceptor site')
        plt.tight_layout()
        plt.savefig(f"{outdir}/calibration/prob_dist_acceptor.png")
    elif index == 2:
        plt.title('Score Distribution for donor site')
        plt.tight_layout()
        plt.savefig(f"{outdir}/calibration/prob_dist_donor.png")
    plt.clf()


def initialize_model_and_optim(device, flanking_size, pretrained_model):
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS    
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    # Initialize the model
    model = SpliceAI(L, W, AR).to(device)
    # Print the shapes of the parameters in the initialized model
    print("\nInitialized model parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", end=", ")

    # Load the pretrained model
    state_dict = torch.load(pretrained_model, map_location=device)

    # Filter out unnecessary keys and load matching keys into model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print missing and unexpected keys
    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, params


def calibrate(args):
    print('Running OpenSpliceAI with calibrate mode.')
    print("\n--------------------------------------------------------------")
    start_time = time.time()
    
    # Setup environment, load datasets, etc.
    device = setup_environment(args)
    train_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(batch_num, args.random_seed, test_h5f)
    model, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model)
    scaled_model = ModelWithTemperature(model)
    print("scaled_model: ", scaled_model)
    # Check if temperature_file is provided
    valid_loader = get_validation_loader(test_h5f, test_idxs, params["BATCH_SIZE"])
    if args.temperature_file is not None:
        # Load the temperature
        scaled_model.load_temperature(args.temperature_file, valid_loader, params)
        print(f"Loaded temperature from {args.temperature_file}: {scaled_model.temperature.item()}")
        end_time = time.time()
        print(f"Time taken to load the calibrated model: {end_time - start_time:.2f} seconds")
    else:
        # Train the temperature
        scaled_model.set_temperature(valid_loader, params)
        # Save the temperature parameter
        temperature_save_path = f"{args.output_dir}/temperature.pt"
        scaled_model.save_temperature(temperature_save_path)
        print(f"Saved temperature to {temperature_save_path}")
        end_time = time.time()
        print(f"Time taken to calibrate the model: {end_time - start_time:.2f} seconds")

    temperature_txt_save_path = f"{args.output_dir}/temperature.txt"
    with open(temperature_txt_save_path, 'w') as f:
        f.write(f"{scaled_model.temperature.item()}\n")
    # Apply temperature scaling
    logits = scaled_model.logits
    labels = scaled_model.labels
    logits_scaled = None
    print("logits: ", logits)
    print("logits: ", logits.shape)
    print("labels: ", labels)
    print("labels: ", labels.shape)

    with torch.no_grad():
        logits_scaled = scaled_model.temperature_scale(logits)

    print("logits_scaled: ", logits_scaled)
    print("logits_scaled: ", logits_scaled.shape)
    print("logits: ", logits)
    print("logits: ", logits.shape)
    print("labels: ", labels)
    print("labels: ", labels.shape)

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1).cpu().numpy()
    probs_scaled = F.softmax(logits_scaled, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    os.makedirs(f'{args.output_dir}/calibration', exist_ok=True)

    # ##########################################
    # # Plotting the score distribution
    # ##########################################
    score_frequency_distribution(probs, probs_scaled, labels, args.output_dir, index=0)
    score_frequency_distribution(probs, probs_scaled, labels, args.output_dir, index=1)
    score_frequency_distribution(probs, probs_scaled, labels, args.output_dir, index=2)

    ##########################################
    # Plotting the calibration curve
    ##########################################
    # Compute the calibration curve for each class 
    classes = ["Non-splice site", "Acceptor site", "Donor site"]
    calibration_curve_values = [] 
    calibration_curve_scaled_values = [] 
    for i in range(3):
        # Filter out the samples for the current class
        class_labels = (labels == i).astype(int)
        class_probs = probs[:, i]
        class_probs_scaled = probs_scaled[:, i]
        print(f"\nClass: {classes[i]}")
        print("\tclass_labels: ", class_labels.shape)
        print("\tclass_probs: ", class_probs.shape)
        print("\tclass_probs_scaled: ", class_probs_scaled.shape)

        # Compute calibration curves with bin counts
        prob_true, prob_pred, bin_counts = compute_calibration_curve(class_labels, class_probs, n_bins=30, strategy='uniform')
        prob_true_scaled, prob_pred_scaled, _ = compute_calibration_curve(class_labels, class_probs_scaled, n_bins=30, strategy='uniform')

        print("\tprob_true: ", prob_true.shape)
        print("\tprob_pred: ", prob_pred.shape)
        print("\tbin_counts: ", bin_counts.shape)
        print("\tprob_true_scaled: ", prob_true_scaled.shape)
        print("\tprob_pred_scaled: ", prob_pred_scaled.shape)
        
        calibration_curve_values.append((prob_true, prob_pred, bin_counts))
        calibration_curve_scaled_values.append((prob_true_scaled, prob_pred_scaled, bin_counts))
        print("===============================================")

    print("calibration_curve_values: ", calibration_curve_values)
    print("calibration_curve_scaled_values: ", calibration_curve_scaled_values)
    # Create a directory to save calibration data
    calibration_data_dir = os.path.join(args.output_dir, 'calibration_data')
    os.makedirs(calibration_data_dir, exist_ok=True)
    flanking_size = args.flanking_size

    # Save the calibration data
    for i in range(3):
        prob_true, prob_pred, bin_counts = calibration_curve_values[i]
        prob_true_scaled, prob_pred_scaled, _ = calibration_curve_scaled_values[i]
        class_name = classes[i]
        # Save original calibration data
        np.savez(
            os.path.join(calibration_data_dir, f'calibration_data_{class_name}_original_{flanking_size}nt.npz'),
            prob_true=prob_true,
            prob_pred=prob_pred,
            bin_counts=bin_counts
        )

        # Save calibrated calibration data
        np.savez(
            os.path.join(calibration_data_dir, f'calibration_data_{class_name}_calibrated_{flanking_size}nt.npz'),
            prob_true=prob_true_scaled,
            prob_pred=prob_pred_scaled,
            bin_counts=bin_counts
        )

    # Plot the calibration curves with confidence intervals and sample sizes
    fig, axs = plt.subplots(1, 3, figsize=(20,5.5)) 
    for i in range(3):
        prob_true, prob_pred, bin_counts = calibration_curve_values[i]
        prob_true_scaled, prob_pred_scaled, _ = calibration_curve_scaled_values[i]

        # Compute confidence intervals
        ci_lower, ci_upper = compute_confidence_intervals(prob_true, bin_counts)
        ci_lower_scaled, ci_upper_scaled = compute_confidence_intervals(prob_true_scaled, bin_counts)

        # Original probabilities
        axs[i].plot(prob_pred, prob_true, marker='o', label='Original probability', color='blue')
        axs[i].fill_between(prob_pred, ci_lower, ci_upper, color='blue', alpha=0.2, label='Confidence Interval (Original)')

        # Calibrated probabilities
        axs[i].plot(prob_pred_scaled, prob_true_scaled, marker='o', label='Calibrated probability', color='green')
        axs[i].fill_between(prob_pred_scaled, ci_lower_scaled, ci_upper_scaled, color='green', alpha=0.1, label='Confidence Interval (Calibrated)')

        # Diagonal line
        axs[i].plot([0, 1], [0, 1], linestyle='--', color='gray') 

        axs[i].set_xlim([0, 1]) 
        axs[i].set_ylim([0, 1]) 
        axs[i].set_title(f"{classes[i]}", fontsize=17) 
        axs[i].set_xlabel("Predicted probability", fontsize=15) 
        axs[i].set_ylabel("Empirical probability", fontsize=15) 
        # # Annotate sample sizes
        # for x, y, count in zip(prob_pred, prob_true, bin_counts):
        #     axs[i].annotate(f'n={count}', xy=(x, y), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
        axs[i].legend()
    plt.tight_layout() 
    plt.savefig(f'{args.output_dir}/calibration/calibration_curve.png', dpi=300)
    plt.clf()

    brier_scores_uncalibrated = []
    brier_scores_calibrated = []

    for i, class_name in enumerate(classes):
        class_labels = (labels == i).astype(int)
        class_probs = probs[:, i]  # Uncalibrated probabilities
        class_probs_calibrated = probs_scaled[:, i]  # Calibrated probabilities

        # Compute Brier Score
        brier_uncal = brier_score_loss(class_labels, class_probs)
        brier_cal = brier_score_loss(class_labels, class_probs_calibrated)

        brier_scores_uncalibrated.append(brier_uncal)
        brier_scores_calibrated.append(brier_cal)

    # Visualization
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, brier_scores_uncalibrated, width, label='Uncalibrated')
    rects2 = ax.bar(x + width/2, brier_scores_calibrated, width, label='Calibrated')

    ax.set_ylabel('Brier Score')
    ax.set_title('Brier Score by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/calibration/brier_scores.png', dpi=300)
    plt.close()



    ##########################################
    # Plotting calibration map
    ##########################################
    # Create a grid of probability values
    p1d = np.linspace(0, 1, 20)
    p0, p1 = np.meshgrid(p1d, p1d)
    p2 = 1 - p0 - p1
    # Keep only valid probabilities (where p2 >= 0)
    mask = (p2 >= 0) & (p2 <= 1)
    p0 = p0[mask]
    p1 = p1[mask]
    p2 = p2[mask]
    p = np.vstack([p0, p1, p2]).T  # Shape (num_points, 3)

    # Convert probabilities to logits
    p_logits = reverse_softmax(p)

    # Apply temperature scaling
    p_logits_tensor = torch.tensor(p_logits, dtype=torch.float32).to(device)
    with torch.no_grad():
        p_logits_scaled = scaled_model.temperature_scale(p_logits_tensor)

    # Convert back to probabilities
    p_probs_scaled = torch.nn.functional.softmax(p_logits_scaled, dim=1)
    p_probs_scaled = p_probs_scaled.detach().cpu().numpy()

    # Plot the calibration map
    plt.figure(figsize=(4.8, 4.8))
    colors = ["red", "green", "blue"]  # Colors for each class
    for i in range(p.shape[0]):
        # Original probabilities
        p_orig = p[i]
        # Calibrated probabilities
        p_new = p_probs_scaled[i]
        # Determine the color based on the class with the highest original probability
        color = colors[np.argmax(p_orig)]
        # Plot an arrow from original to calibrated probability
        plt.arrow(
            p_orig[1], p_orig[2],
            p_new[1] - p_orig[1], p_new[2] - p_orig[2],
            color=color, head_width=0.01, length_includes_head=True
        )

    # Plot the simplex boundaries
    plt.plot([0, 1], [0, 0], 'k-')  # Bottom edge
    plt.plot([1, 0], [0, 1], 'k-')  # Right edge
    plt.plot([0, 0], [1, 0], 'k-')  # Left edge

    # Set labels and title
    plt.xlabel("Probability of Acceptor site (Class 1)")
    plt.ylabel("Probability of Donor site (Class 2)")
    plt.title("Calibration Map with Temperature Scaling")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(handles=[
        plt.Line2D([0], [0], color='red', lw=2, label='Non-splice site'),
        plt.Line2D([0], [0], color='green', lw=2, label='Acceptor site'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Donor site')
    ])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/calibration/temperature_scaling_calibration_map.png", dpi=300)
    plt.clf()

    train_h5f.close()
    test_h5f.close()
