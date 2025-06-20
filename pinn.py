import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # ensures output between 0 and 1
        )

    def forward(self, x):
        return self.net(x)


def pinn_loss(preds, y_true, x_batch, F_idx, D_idx, B_idx, lambda_mass=1000.0, lambda_bounds=100.0):
    mse = nn.MSELoss()
    data_loss = mse(preds, y_true)

    # Physics-informed mass balance loss: F ≈ D + B
    mass_loss = mse(x_batch[:, F_idx], x_batch[:, D_idx] + x_batch[:, B_idx])

    # Bounds penalty loss (should already be [0,1] from sigmoid)
    bounds_loss = mse(torch.clamp(preds, 0, 1), preds)

    total_loss = data_loss + lambda_mass * mass_loss + lambda_bounds * bounds_loss
    return total_loss, data_loss.item(), mass_loss.item(), bounds_loss.item()
