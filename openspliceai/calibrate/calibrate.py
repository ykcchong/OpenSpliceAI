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
    plt.figure(figsize=(10, 6))
    # Probability distribution for Class 1 from logits (only positive cases)
    plt.hist(probabilities_positive, bins=50, alpha=0.5, label='Predicted Probabilities')
    plt.hist(probabilities_scaled_positive, bins=50, alpha=0.5, label='Predicted Calibrated Probabilities')
    # For positive cases, since all labels are 1, we can simply indicate the average or density with a line or annotation
    # plt.axvline(x=np.mean(labels_positive), color='r', linestyle='dashed', linewidth=2, label='Actual Labels')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    if index == 0:
        plt.title('Score Distribution for non-splice site')
        plt.savefig(f"{outdir}/calibration/prob_dist_neither.png")
    if index == 1:
        plt.title('Score Distribution for acceptor site')
        plt.savefig(f"{outdir}/calibration/prob_dist_acceptor.png")
    elif index == 2:
        plt.title('Score Distribution for donor site')
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
    valid_loader = get_validation_loader(test_h5f, test_idxs, params["BATCH_SIZE"])
    scaled_model.set_temperature(valid_loader, params)
    # Save the scaled model
    torch.save(scaled_model.state_dict(), f"{args.output_dir}/temp_model.pt")
    print("scaled_model: ", scaled_model)
    end_time = time.time()
    print(f"Time taken to calibrate the model: {end_time - start_time:.2f} seconds")

    # Apply temperature scaling
    logits = scaled_model.logits
    labels = scaled_model.labels
    logits_scaled = None
    with torch.no_grad():
        logits_scaled = scaled_model.temperature_scale(logits)

    print("logits_scaled: ", logits_scaled)
    print("logits_scaled: ", logits_scaled.shape)
    print("logits: ", logits)
    print("logits: ", logits.shape)
    print("labels: ", labels)
    print("labels: ", labels.shape)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs_scaled = torch.nn.functional.softmax(logits_scaled, dim=1)
    print("probs: ", probs)
    print("probs: ", probs.shape)
    print("probs_scaled: ", probs_scaled)
    print("probs_scaled: ", probs_scaled.shape)

    # Convert tensors to CPU before converting to NumPy for use with sklearn's log_loss
    probs = probs.detach().cpu().numpy()
    probs_scaled = probs_scaled.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
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
    from sklearn.calibration import calibration_curve 
    calibration_curve_values = [] 
    calibration_curve_scaled_values = [] 
    for i in range(3): 
        curve = calibration_curve(labels == i,  
                                probs[:, i],  
                                n_bins=30,  
                                pos_label=True)
        scaled_curve = calibration_curve(labels == i,  
                                probs_scaled[:, i],  
                                n_bins=30,  
                                pos_label=True)  
        calibration_curve_values.append(curve)
        calibration_curve_scaled_values.append(scaled_curve) 
    
    # Plot the calibration curves 
    fig, axs = plt.subplots(1, 3, figsize=(17,5)) 
    for i in range(3): 
        axs[i].plot(calibration_curve_values[i][1],  
                    calibration_curve_values[i][0],  
                    marker='o', label='Original probability') 
        axs[i].plot(calibration_curve_scaled_values[i][1],  
                    calibration_curve_scaled_values[i][0],  
                    marker='o', label='Calibrated probability') 
        axs[i].plot([0, 1], [0, 1], linestyle='--') 
        axs[i].set_xlim([0, 1]) 
        axs[i].set_ylim([0, 1]) 
        axs[i].set_title(f"{classes[i]}", fontsize = 17) 
        axs[i].set_xlabel("Predicted probability", fontsize = 15) 
        axs[i].set_ylabel("Empirical probability", fontsize = 15) 
        axs[i].legend()
    plt.tight_layout() 
    plt.savefig(f'{args.output_dir}/calibration/calibration_curve.png')
    plt.clf()
    train_h5f.close()
    test_h5f.close()
