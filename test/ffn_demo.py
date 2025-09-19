import math
import random
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn


class FeedForwardNetwork(nn.Module):
    """A small fully-connected feedforward neural network for regression."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_synthetic_regression_data(
    num_samples: int = 2000, noise_std: float = 0.1
) -> Tuple[Tensor, Tensor]:
    """
    Create a simple non-linear regression dataset.

    Features: x1, x2 sampled from N(0, 1)
    Target: y = sin(x1) + 0.5 * x2^2 + noise
    """
    x = np.random.randn(num_samples, 2).astype(np.float32)
    x1 = x[:, 0]
    x2 = x[:, 1]
    y = np.sin(x1) + 0.5 * (x2 ** 2)
    y += np.random.randn(num_samples).astype(np.float32) * noise_std
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y).unsqueeze(1).float()
    return x_tensor, y_tensor


def train(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    num_samples = train_x.shape[0]
    num_batches = math.ceil(num_samples / batch_size)

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            indices = permutation[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_x = train_x[indices]
            batch_y = train_y[indices]

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_mse = epoch_loss / num_samples
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE: {epoch_mse:.6f}")


@torch.no_grad()
def evaluate(model: nn.Module, test_x: Tensor, test_y: Tensor) -> float:
    device = next(model.parameters()).device
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    predictions = model(test_x)
    mse = nn.functional.mse_loss(predictions, test_y).item()
    return mse


def main() -> None:
    set_seed(42)

    # Generate data and split into train/test
    features, targets = generate_synthetic_regression_data(num_samples=3000, noise_std=0.15)
    num_train = int(0.8 * features.size(0))
    train_x, test_x = features[:num_train], features[num_train:]
    train_y, test_y = targets[:num_train], targets[num_train:]

    # Build model
    model = FeedForwardNetwork(input_dim=2, hidden_dim=64, output_dim=1)

    # Train
    train(
        model=model,
        train_x=train_x,
        train_y=train_y,
        epochs=200,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )

    # Evaluate
    test_mse = evaluate(model, test_x, test_y)
    print(f"Test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()


