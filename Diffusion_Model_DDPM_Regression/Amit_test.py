import pandas as pd
import torch
import numpy as np
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

n_epochs = 5000  #backward training step
x_dim = 4  #dimension of the independence variables

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
    
# Path to your saved models
best_cond_pred_model_path = 'best_cond_pred_model.pth'  # Path to the saved conditional prediction model
best_diff_model_path = 'best_diff_model.pth'  # Path to the saved diffusion model

# Load the CSV file
csv_file = 'bayern.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Extract the 'biomass' and 'LAI' columns as inputs
X_data = data[['RainYearly', 'RadiationYearly', 'TmaxYearly', 'TminYearly']].values  # Ensure column names match your CSV
Y = data['Obs_Yield_t_ha'].values

# Normalize the inputs (if necessary)
# If the model expects normalized inputs, normalize using the same mean and std as during training
# For demonstration, we assume the mean and std are known
X_mean, X_std = X_data.mean(axis=0), X_data.std(axis=0)

y_mean, y_std = Y.mean(), Y.std()

X_normalized = (X_data - X_mean) / X_std

# Convert to torch tensors
X_test = torch.tensor(X_normalized, dtype=torch.float32)

cat_x = True
cat_y_pred = True

y_dim = 1
z_dim = 2
n_steps = 1000   #forward training step
hid_layers = [100, 50]

beta_schedule = "linear"
beta_start = 0.0001
beta_end = 0.02

n_z_samples = 100

cond_pred_model = DeterministicFeedForwardNeuralNetwork(dim_in=x_dim,
                                                        dim_out=y_dim,
                                                        hid_layers=hid_layers
                                                        )
diff_model = ConditionalGuidedModel(
    n_steps=n_steps,
    cat_x=cat_x,
    cat_y_pred=cat_y_pred,
    x_dim=x_dim,
    y_dim=y_dim,
    z_dim=z_dim,
)

# Load the state_dict into the models
cond_pred_model.load_state_dict(torch.load(best_cond_pred_model_path, map_location=torch.device(device)))
diff_model.load_state_dict(torch.load(best_diff_model_path, map_location=torch.device(device)))

# Move the models to the correct device
cond_pred_model = cond_pred_model.to(device)
diff_model = diff_model.to(device)

# Move data to device
X_test = X_test.to(device)

# Set the models to evaluation mode
cond_pred_model.eval()
diff_model.eval()

# Predict y using the conditional model
with torch.no_grad():
    y_0_hat = cond_pred_model(X_test)

# Tile y_0_hat and X_test for reverse diffusion sampling
n_z_samples = 100  # Number of samples per input
y_0_hat_tile = y_0_hat.repeat_interleave(n_z_samples, dim=0)
test_x_tile = X_test.repeat_interleave(n_z_samples, dim=0)

# Add noise to y_0_hat_tile
z = torch.randn_like(y_0_hat_tile).to(device)
y_t = y_0_hat_tile + z

betas = make_beta_schedule(beta_schedule, n_steps, beta_start, beta_end).to(device)
betas_sqrt = torch.sqrt(betas)
alphas = 1.0 - betas
alphas_cumprod = alphas.cumprod(dim=0)
alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

# Perform reverse diffusion sampling
y_tile_seq = p_sample_loop(
    diff_model,
    test_x_tile,
    y_0_hat_tile,
    y_t,
    n_steps,
    alphas,
    one_minus_alphas_bar_sqrt,
)

# Extract final predictions
# Reshape to (num_samples, n_z_samples)
num_samples = X_test.shape[0]
y_samples = y_tile_seq[-1].reshape(num_samples, n_z_samples)

# Compute the mean prediction for each sample
y_pred_mean = y_samples.mean(dim=1)

# If the target variable was normalized during training, unnormalize it
# Assume y_mean and y_std are known from training data normalization
# Replace y_mean and y_std with your actual values
y_pred = y_pred_mean.cpu().numpy()
y_pred = y_pred * y_std + y_mean  # Unnormalize predictions

# Save the predictions to a CSV file
output_df = pd.DataFrame({
    'predicted_y': y_pred
})

output_df.to_csv('predicted_y.csv', index=False)
print("Predictions saved to 'predicted_y.csv'")