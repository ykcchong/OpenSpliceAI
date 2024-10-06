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
        # Move temperature to the device of logits
        temperature = torch.clamp(self.temperature, min=0.05, max=5.0).to(logits.device)
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
