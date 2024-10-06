import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from openspliceai.train_base.utils import *
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def load_data_from_shard(h5f, shard_idx):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    return X, Y

def get_validation_loader(h5f, idxs, batch_size):
    """
    Create a DataLoader for the validation data.
    """
    X_list = []
    Y_list = []
    for shard_idx in idxs:
        X, Y = load_data_from_shard(h5f, shard_idx)
        X_list.append(X)
        Y_list.append(Y)
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print("X: ", X.shape)   
    print("Y: ", Y.shape)
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


class ModelWithTemperature(nn.Module):
    """
    A decorator class that wraps a model with temperature scaling.
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.temperature.data = torch.clamp(self.temperature.data, min=0.05, max=5.0)
        self.logits = None
        self.labels = None

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits.
        """
        temperature = torch.clamp(self.temperature, min=0.05, max=5.0)
        return logits / temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))



    def set_temperature(self, valid_loader, params):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(valid_loader, desc='Collecting logits and labels'):
                input = input.to(device)
                label = label.to(device)
                input, label = clip_datapoints(input, label, params["CL"], CL_max, params["N_GPUS"])
                logits = self.model(input)
                # logits_list.append(logits)
                # labels_list.append(label)
                logits_list.append(logits.detach().cpu())
                labels_list.append(label.detach().cpu())
        logits = torch.cat(logits_list)  # Shape: (N, C, L)
        labels = torch.cat(labels_list)  # Shape: (N, C, L)

        # Permute logits to (N, L, C)
        logits = logits.permute(0, 2, 1).contiguous()  # Shape: (N, L, C)
        # Permute labels to (N, L, C)
        labels = labels.permute(0, 2, 1).contiguous()  # Shape: (N, L, C)

        # Convert labels from one-hot to class indices
        labels = labels.argmax(dim=-1)  # Shape: (N, L)

        # Flatten logits and labels
        N, L, C = logits.shape
        logits = logits.view(-1, C)      # Shape: (N * L, C)
        labels = labels.view(-1).long()  # Shape: (N * L,)

        # Store logits and labels as attributes for later use
        self.logits = logits
        self.labels = labels
        
        # Move to device (if not already on CUDA)
        logits = logits.to(device)
        labels = labels.to(device)

        # Compute NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.4f, ECE: %.4f' % (before_temperature_nll, before_temperature_ece))

        # Optimize temperature using LBFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        # Add a learning rate scheduler (if needed)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        
        def eval():        
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_([self.temperature], max_norm=1.0)  # You can choose an appropriate max_norm value

            return loss

        print(f"Initial temperature: {self.temperature.item()}")
        optimizer.step(eval)
        # scheduler.step(eval)

        # # Iteratively optimize and step scheduler
        # for step in range(50):  # Adjust the number of steps as needed
        #     optimizer.step(eval)
            
        #     # Compute loss after the optimization step
        #     current_loss = nll_criterion(self.temperature_scale(logits), labels).item()
        #     print(f"Step {step}: Current NLL loss: {current_loss}")
            
        #     # Step the scheduler
        #     scheduler.step(current_loss)

        print(f"Optimized temperature: {self.temperature.item()}")
        # Compute NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.4f' % self.temperature.item())
        print('After temperature - NLL: %.4f, ECE: %.4f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): Number of confidence interval bins.
        """
        super(_ECELoss, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(device)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculate |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

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
    # device = setup_environment(args)
    # # model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    # train_h5f, test_h5f, batch_num = load_datasets(args)
    # train_idxs, val_idxs, test_idxs = generate_indices(batch_num, args.random_seed, test_h5f)
    # model, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model)
    

    print("\n--------------------------------------------------------------")
    start_time = time.time()
    # ##########################################
    # # Temperature scaling
    # ##########################################    
    # scaled_model = ModelWithTemperature(model)
    # scaled_model.set_temperature(model, test_h5f, test_idxs, device, params["BATCH_SIZE"], params, train=True)
    # # scaled_model = ModelWithTemperature(model).to(device)
    # # scaled_model.load_state_dict(torch.load("temp_model.pt"))
    # # scaled_model = scaled_model.to(device)
    # # scaled_model.set_temperature(model, train_h5f, val_idxs, device, params["BATCH_SIZE"], params, train=False)
    # # print("logits: ", logits.shape)
    # # print("labels: ", labels.shape)
    # torch.save(scaled_model.state_dict(), f"temp_model.pt")


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
    # # Assuming labels are one-hot encoded, we convert them to class indices for log_loss
    # # Calculate log-loss
    # log_loss_before = log_loss(labels, probs)
    # log_loss_after = log_loss(labels, probs_scaled)
    # # print("probs_after_reshaped: ", probs_after_reshaped)
    # # print("labels_indices: ", labels_indices)
    # print("Log-loss of")
    # print(f" * uncalibrated classifier: {log_loss_before:.8f}")
    # print(f" * calibrated classifier: {log_loss_after:.8f}")


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

    # params["RANDOM_SEED"] = args.random_seed
    # train_metric_files = create_metric_files(log_output_train_base)
    # valid_metric_files = create_metric_files(log_output_val_base)
    # test_metric_files = create_metric_files(log_output_test_base)
    # train_model(model, optimizer, scheduler, train_h5f, test_h5f, train_idxs, val_idxs, test_idxs, 
    #             model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
    train_h5f.close()
    test_h5f.close()
