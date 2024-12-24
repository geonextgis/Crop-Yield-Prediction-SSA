import time
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
from tqdm import trange
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import r2_score
from lightning import LightningDataModule
from typing import Union, Callable, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import (
    DeterministicFeedForwardNeuralNetwork,
    ConditionalLinear,
)
from diffusion_utils import (
    make_beta_schedule,
    q_sample,
    p_sample_loop,
    p_sample,
)

n_pretrain_epochs = 1000 #forward training step
n_epochs = 5000  #backward training step
x_dim = 4  #dimension of the independence variables


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

sns.set(style="darkgrid", palette="husl")

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class CSVDataModule:
    def __init__(self, csv_file: str, batch_size: int = 64, test_size: float = 0.2, random_state: int = 42):
        # Load the data from the CSV file
        self.data = pd.read_csv(csv_file)

        # Extract the X (features) and Y (target)
        self.X = self.data[['RainYearly', 'RadiationYearly', 'TmaxYearly', 'TminYearly']].values  # Features: 'RainYearly', 'RadiationYearly', 'TmaxYearly', 'TminYearly'
        self.Y = self.data['Obs_Yield_t_ha'].values  # Target: 'yield'

        # Normalize the data (optional)
        self.X_mean, self.X_std = self.X.mean(axis=0), self.X.std(axis=0)
        self.Y_mean, self.Y_std = self.Y.mean(), self.Y.std()
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        self.Y_normalized = (self.Y - self.Y_mean) / self.Y_std

        # Convert to PyTorch tensors
        self.X_tensor = torch.tensor(self.X_normalized, dtype=torch.float32)
        self.Y_tensor = torch.tensor(self.Y_normalized, dtype=torch.float32).unsqueeze(-1)

        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_tensor, self.Y_tensor, test_size=test_size, random_state=random_state)

        self.batch_size = batch_size

    def prepare_data(self):
        """Prepare training and validation data loaders."""
        self.train_data = TensorDataset(self.X_train, self.y_train)
        self.test_data = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def get_train_data(self):
        """Return training data."""
        return self.X_train, self.y_train

    def get_test_data(self):
        """Return test data."""
        return self.X_test, self.y_test


class ConditionalGuidedModel(nn.Module):
    def __init__(
        self,
        n_steps: int,
        cat_x: bool,
        cat_y_pred: bool,
        x_dim: int,
        y_dim: int,
        z_dim: int,
    ):
        super(ConditionalGuidedModel, self).__init__()
        self.cat_x = cat_x
        self.cat_y_pred = cat_y_pred
        data_dim = y_dim
        if self.cat_x:
            data_dim += x_dim
        if self.cat_y_pred:
            data_dim += y_dim
        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 1)

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        return self.lin4(eps_pred)
    
    
# Example usage
csv_file = "bayern.csv"
data_module = CSVDataModule(csv_file)
data_module.prepare_data()

train_loader = data_module.train_dataloader()
test_loader = data_module.test_dataloader()

# Optionally, access the raw train/test data
X_train, y_train = data_module.get_train_data()
X_test, y_test = data_module.get_test_data()

cat_x = True
cat_y_pred = True

y_dim = 1
z_dim = 2

hid_layers = [100, 50]

beta_schedule = "linear"
beta_start = 0.0001
beta_end = 0.02
n_steps = 1000   #diffusion step
n_z_samples = 100

diff_model = ConditionalGuidedModel(
    n_steps=n_steps,
    cat_x=cat_x,
    cat_y_pred=cat_y_pred,
    x_dim=x_dim,
    y_dim=y_dim,
    z_dim=z_dim,
)
diff_model.to(device)

cond_pred_model = DeterministicFeedForwardNeuralNetwork(dim_in=x_dim,
                                                        dim_out=y_dim,
                                                        hid_layers=hid_layers
                                                        )
cond_pred_model.to(device)


aux_optimizer = Adam(cond_pred_model.parameters(), lr=0.001)
aux_cost_fn = nn.MSELoss()
cond_pred_model.train()

# Lists to store losses
train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize with a large value
best_model_path = 'best_cond_pred_model.pth'  # Path to save the best model

# Training loop
bar = trange(n_pretrain_epochs, leave=True)
for epoch in bar:
    cond_pred_model.train()  # Set model to training mode
    running_train_loss = 0.0

    # Training step
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_pred = cond_pred_model(x)
        aux_cost = aux_cost_fn(y_pred, y)

        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()

        running_train_loss += aux_cost.item()

    # Average training loss for the epoch
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step (optional, assuming you have a validation set)
    cond_pred_model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            y_val_pred = cond_pred_model(x_val)
            val_cost = aux_cost_fn(y_val_pred, y_val)
            running_val_loss += val_cost.item()

    # Average validation loss for the epoch
    avg_val_loss = running_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    # Check if the validation loss is the best and save the model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(cond_pred_model.state_dict(), best_model_path)  # Save the best model
        print(f"Epoch {epoch + 1}: New best model saved with validation loss {best_val_loss:.4f}")

    bar.set_description(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    

betas = make_beta_schedule(beta_schedule, n_steps, beta_start, beta_end).to(device)
betas_sqrt = torch.sqrt(betas)
alphas = 1.0 - betas
alphas_cumprod = alphas.cumprod(dim=0)
alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)


optimizer = Adam(diff_model.parameters(), lr=0.01)

# Optimizer for cond_pred_model (non-linear guidance model)
aux_optimizer = Adam(cond_pred_model.parameters(), lr=0.001)

# Loss function for cond_pred_model
aux_cost_fn = nn.MSELoss()

# Tracking best loss (initialize best test loss as a large value)
best_test_loss = float('inf')
best_model_path = "best_diff_model.pth"

# Lists to store loss values
train_losses = []
val_losses = []
test_losses = []

diff_bar = trange(n_epochs, leave=True)
diff_model.train()
cond_pred_model.train()

for epoch in diff_bar:
    running_train_loss = 0.0
    running_val_loss = 0.0
    running_test_loss = 0.0

    # Training step
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]

        # Antithetic sampling
        ant_samples_t = torch.randint(low=0, high=n_steps, size=(batch_size // 2 + 1,)).to(device)
        ant_samples_t = torch.cat([ant_samples_t, n_steps - 1 - ant_samples_t], dim=0)[:batch_size]

        # Noise estimation loss
        y_0_hat = cond_pred_model(x)

        e = torch.randn_like(y)

        y_t_sample = q_sample(
            y,
            y_0_hat,
            alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt,
            ant_samples_t,
            noise=e,
        )

        model_output = diff_model(x, y_t_sample, y_0_hat, ant_samples_t)

        # Compute the loss using the same noise sample e
        loss = (e - model_output).square().mean()

        # Optimize diffusion model (diff_model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optimize non-linear guidance model (cond_pred_model)
        aux_cost = aux_cost_fn(cond_pred_model(x), y)
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()

        running_train_loss += loss.item()

    # Average training loss for the epoch
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    # Test step: calculate test loss for each epoch
    running_test_loss = 0.0
    with torch.no_grad():
        for x_test, y_test in test_loader:

            x_test, y_test = x_test.to(device), y_test.to(device)
            y_0_hat_test = cond_pred_model(x_test)
            test_e = torch.randn_like(y_test)

            y_t_test_sample = q_sample(
                y_test,
                y_0_hat_test,
                alphas_bar_sqrt,
                one_minus_alphas_bar_sqrt,
                ant_samples_t,
                noise=test_e
            )

            test_model_output = diff_model(x_test, y_t_test_sample, y_0_hat_test, ant_samples_t)
            test_loss = (test_e - test_model_output).square().mean()
            running_test_loss += test_loss.item()

    avg_test_loss = running_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Set models back to training mode after validation and testing
    cond_pred_model.train()
    diff_model.train()

    # Check if it's the best test loss and save the model
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(diff_model.state_dict(), best_model_path)
        print(f"Epoch {epoch + 1}: New best model saved with test loss {best_test_loss:.4f}")

    diff_bar.set_description(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}", refresh=True)

