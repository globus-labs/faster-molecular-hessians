"""Create a Gaussian-process-regression based model which uses SOAP features"""
import pandas as pd
from ase.calculators.calculator import Calculator, all_changes
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

    Args:
        batch_x: Example input batch of descriptors
        inducing_x: Starting points for the reference points of the kernel
        likelihood: Likelihood function used to describe the noise
        use_ard: Whether to employ a different length scale parameter for each descriptor,
            a technique known as Automatic Relevance Detection (ARD)
    """

    def __init__(self, batch_x: torch.Tensor, inducing_x: torch.Tensor, likelihood: GaussianLikelihood, use_ard: bool):
        batch_y = torch.zeros(batch_x.shape[0], dtype=batch_x.dtype)
        super().__init__(batch_x, batch_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel(has_lengthscale=True, ard_num_dims=batch_x.shape[-1] if use_ard else None))
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_x.clone(), likelihood=likelihood)

    def forward(self, x) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def make_gpr_model(train_descriptors: np.ndarray, num_inducing_points: int, use_ard_kernel: bool = False) -> InducedKernelGPR:
    """Make the GPR model for a certain atomic system

    Args:
        train_descriptors: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        num_inducing_points: Number of inducing points to use in the kernel. More points, more complex model
        use_ard_kernel: Whether to use a different length scale parameter for each descriptor
    Returns:
        Model which can predict energy given descriptors for a single configuration
    """

    # Select a set of inducing points randomly
    descriptors = np.reshape(train_descriptors, (-1, train_descriptors.shape[-1]))
    inducing_inds = np.random.choice(descriptors.shape[0], size=(num_inducing_points,), replace=False)
    inducing_points = descriptors[inducing_inds, :]

    # Make the model
    #  TODO (wardlt): Consider making a batch size of more than one. We currently use the entire training set in a single batch
    return InducedKernelGPR(
        batch_x=torch.from_numpy(descriptors),
        inducing_x=torch.from_numpy(inducing_points),
        use_ard=use_ard_kernel,
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
        Mean loss over each iteration
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
        # TODO (wardlt): Provide training data in batches if it becomes too large
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


class SOAPCalculator(Calculator):
    """Calculator which uses a GPR model trained using SOAP descriptors

    Args:
        model (InducedKernelGPR): A machine learning model which takes descriptors as inputs
        soap (SOAP): Tool used to compute the descriptors
        scaling (tuple[float, float]): A offset and factor with which to adjust the energy per atom predictions,
            which are typically he mean and standard deviation of energy per atom across the training set.
    """

    implemented_properties = ['energy', 'forces', 'energies']
    default_parameters = {
        'model': None,
        'soap': None,
        'scaling': (0., 1.)
    }

    def calculate(self, atoms=None, properties=('energy', 'forces', 'energies'),
                  system_changes=all_changes):
        # Compute the descriptors for the atoms
        d_desc_d_pos, desc = self.parameters['soap'].derivatives(atoms, attach=True)
        desc = torch.from_numpy(desc)
        desc.requires_grad = True
        d_desc_d_pos = torch.from_numpy(d_desc_d_pos)

        # Run inference
        offset, scale = self.parameters['scaling']
        model: InducedKernelGPR = self.parameters['model']
        model.eval()  # Ensure we're in eval mode
        pred_energies_dist = model(desc)
        pred_energies = pred_energies_dist.mean
        pred_energy = torch.sum(pred_energies) * scale + offset

        # Compute the forces
        #  See: https://singroup.github.io/dscribe/latest/tutorials/machine_learning/forces_and_energies.html
        # Derivatives for the descriptors are for each center (which is the input to the model) with respect to each atomic coordinate changing.
        # Energy is summed over the contributions from each center.
        # The total force is therefore a sum over the effect of an atom moving on all centers
        d_energy_d_desc = torch.autograd.grad(
            outputs=pred_energy,
            inputs=desc,
            grad_outputs=torch.ones_like(pred_energy),
        )[0]  # Derivative of the energy with respect to the descriptors for each center
        d_energy_d_center_d_pos = torch.einsum('ijkl,il->ijk', d_desc_d_pos, d_energy_d_desc)  # Derivative for each center with respect to each atom
        pred_forces = -d_energy_d_center_d_pos.sum(dim=0) * scale  # Total effect on each center from each atom

        # Store the results
        self.results['forces'] = pred_forces.detach().numpy()
        self.results['energy'] = pred_energy.detach().numpy()
        self.results['energies'] = pred_energy.detach().numpy()
