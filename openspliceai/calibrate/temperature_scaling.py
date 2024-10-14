import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from openspliceai.train_base.utils import *
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


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
    Wraps a model with temperature scaling for better calibration.
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.temperature.data = torch.clamp(self.temperature.data, min=0.05, max=5.0)
        self.logits = None
        self.labels = None

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Apply temperature scaling to the logits.
        """
        temperature = torch.clamp(self.temperature, min=0.05, max=5.0).to(logits.device)
        return logits / temperature.unsqueeze(1).expand_as(logits)

    def collect_labels(self, valid_loader):
        """
        Collect labels from validation loader.
        """
        labels_list = [label.permute(0, 2, 1).contiguous().view(-1).cpu().numpy() for _, label in valid_loader]
        return np.concatenate(labels_list)


    def set_temperature(self, valid_loader, params):
        print("Setting temperature using valid_loader")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # Collect class weights
        class_weights = torch.tensor([10000.0, 10000.0, 1.0], dtype=torch.float32).to(device)

        nll_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        ece_criterion = _ECELoss().to(device)

        # Collect logits and labels
        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in tqdm(valid_loader, desc='Collecting logits and labels'):
                input, label = input.to(device), label.to(device)
                input, label = clip_datapoints(input, label, params["CL"], CL_max, params["N_GPUS"])
                logits = self.model(input)
                logits_list.append(logits.detach().cpu())
                labels_list.append(label.detach().cpu())

        logits = torch.cat(logits_list).permute(0, 2, 1).contiguous()
        labels = torch.cat(labels_list).permute(0, 2, 1).contiguous().argmax(dim=-1)

        # Flatten logits and labels
        N, L, C = logits.shape
        logits = logits.view(-1, C).to(device)
        labels = labels.view(-1).long().to(device)

        self.logits, self.labels = logits, labels

        # Compute NLL and ECE before temperature scaling
        self._compute_and_log_metrics(nll_criterion, ece_criterion, 'Before temperature')

        print(f"Initial temperature: {self.temperature.item()}")
        # Set up the optimizer and learning rate scheduler
        optimizer = optim.Adam([self.temperature], lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

        # Early stopping parameters
        best_loss = float('inf')
        patience = 20  # You can adjust this value
        early_stop_counter = 0
        max_steps = 500  # Increase the number of steps as needed

        for step in range(max_steps):
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            optimizer.step()

            # Check for early stopping
            current_loss = loss.item()
            print(f"Step {step}: NLL loss = {current_loss:.4f}")

            if current_loss < best_loss:
                best_loss = current_loss
                early_stop_counter = 0  # Reset counter if there's improvement
            else:
                early_stop_counter += 1

            # Apply the learning rate scheduler
            scheduler.step(current_loss)

            if early_stop_counter >= patience:
                print(f"Early stopping at step {step}. Best loss: {best_loss:.4f}")
                break

        # Final optimization using LBFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        self._optimize_with_lbfgs(optimizer, nll_criterion, logits, labels)

        # Compute NLL and ECE after temperature scaling
        self._compute_and_log_metrics(nll_criterion, ece_criterion, 'After temperature')

        return self

    def _optimize_temperature(self, nll_criterion, logits, labels, optimizer_cls, lr=0.01, steps=100):
        """
        Helper function to optimize temperature using a given optimizer.
        """
        optimizer = optimizer_cls([self.temperature], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            optimizer.step()

    def _optimize_with_lbfgs(self, optimizer, nll_criterion, logits, labels):
        """
        Helper function to optimize temperature using LBFGS.
        """
        def eval_fn():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.temperature], max_norm=1.0)
            return loss

        optimizer.step(eval_fn)
        print(f"Optimized temperature: {self.temperature.item()}")

    def _compute_and_log_metrics(self, nll_criterion, ece_criterion, phase):
        """
        Helper function to compute and log NLL and ECE metrics.
        """
        nll = nll_criterion(self.logits, self.labels).item()
        ece = ece_criterion(self.logits, self.labels).item()
        print(f'{phase} - NLL: {nll:.4f}, ECE: {ece:.4f}')


class _ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE) Loss.
    """
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.bin_lowers, self.bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece