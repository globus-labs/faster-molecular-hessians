"""Create a Gaussian-process-regression based model which uses SOAP features"""
import pandas as pd
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
import numpy as np
import torch


class InducedKernelGPR(ExactGP):
    """Gaussian process regression model with an induced kernel

    Predicts the energy for each atom as a function of its descriptors
    """

    def __init__(self, batch_x: torch.Tensor, batch_y: torch.Tensor, inducing_x: torch.Tensor, likelihood: GaussianLikelihood, use_adr: bool):
        super().__init__(batch_x, batch_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel(has_lengthscale=True, ard_num_dims=batch_x.shape[-1] if use_adr else None))
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_x.clone(), likelihood=likelihood)

    def forward(self, x) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def make_gpr_model(train_descriptors: np.ndarray, num_inducing_points: int, use_adr_kernel: bool = False) -> InducedKernelGPR:
    """Make the GPR model for a certain atomic system

    Args:
        train_descriptors: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        num_inducing_points: Number of inducing points to use in the kernel. More points, more complex model
        use_adr_kernel: Whether to use a different length scale parameter for each descriptor
    Returns:
        Model which can predict energy given descriptors for a single configuration
    """

    # Select a set of inducing points randomly
    descriptors = np.reshape(train_descriptors, (-1, train_descriptors.shape[-1]))
    inducing_inds = np.random.choice(descriptors.shape[0], size=(num_inducing_points,), replace=False)
    inducing_points = descriptors[inducing_inds, :]

    # Make the model
    #  TODO (wardlt): Consider making a batch size of more than one. Do we do that by just supplying all of the training data
    batch_y = torch.zeros(descriptors.shape[0])  # Use a mean of zero for the training points
    return InducedKernelGPR(
        batch_x=torch.from_numpy(descriptors),
        batch_y=batch_y,  # Returns a scalar for each point
        inducing_x=torch.from_numpy(inducing_points),
        use_adr=use_adr_kernel,
        likelihood=GaussianLikelihood()
    )


def train_model(model: InducedKernelGPR, train_x: np.ndarray, train_y: np.ndarray, steps: int, learning_rate: float = 0.01) -> pd.DataFrame:
    """Train the kernel model over many iterations

    Args:
        model: Model to be trained
        train_x: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        train_y: Energies for each training point
        steps: Number of interactions over all training points
        learning_rate: Learning rate used for the optimizer
    Returns:
        Mean loss over all iterations
    """

    # Convert the data to numpy
    n_conf, n_atoms = train_x.shape[:2]
    train_x = torch.from_numpy(train_x.reshape(-1, train_x.shape[-1]))
    train_y = torch.from_numpy(train_y)

    # Convert the model to training model
    model.train()
    model.likelihood.train()

    # Define the optimizer and loss function
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Iterate over the data multiple times
    losses = []
    for epoch in range(steps):
        # Predict on all configurations
        pred_y_per_atoms_flat = model(train_x)

        # Get the mean sum for each atom
        pred_y_per_atoms = torch.reshape(pred_y_per_atoms_flat.mean, (n_conf, n_atoms))
        pred_y_mean = torch.sum(pred_y_per_atoms, dim=1)

        # The covariance matrix of those sums, assuming they are uncorrelated with each other (they are not)
        pred_y_covar_flat = pred_y_per_atoms_flat.covariance_matrix
        pred_y_covar_grouped_by_conf = pred_y_covar_flat.reshape((n_conf, n_atoms, n_conf, n_atoms))
        pred_y_covar = torch.sum(pred_y_covar_grouped_by_conf, dim=(1, 3))
        pred_y_covar = torch.diag(torch.diag(pred_y_covar))  # Make it diagonal

        # Turn them in to a distribution, and use that to compute a loss function
        pred_y_dist = MultivariateNormal(mean=pred_y_mean, covariance_matrix=pred_y_covar)

        loss = -mll(pred_y_dist, train_y)
        loss.backward()
        losses.append(loss.detach().numpy())
        opt.step()

    return pd.DataFrame({'loss': losses})
