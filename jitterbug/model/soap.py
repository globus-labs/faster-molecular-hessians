"""Create a Gaussian-process-regression based model which uses SOAP features"""
from ase.calculators.calculator import Calculator, all_changes
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


class InducedKernelGPR(ApproximateGP):
    """Gaussian process regression model with an induced kernel

    Predicts the energy for each atom as a function of its descriptors

    Args:
        batch_x: Example input batch of descriptors
        inducing_x: Starting points for the reference points of the kernel
        likelihood: Likelihood function used to describe the noise
        use_ard: Whether to employ a different length scale parameter for each descriptor,
            a technique known as Automatic Relevance Detection (ARD)
    """

    def __init__(self, batch_x: torch.Tensor, inducing_x: torch.Tensor, use_ard: bool):
        variational_distribution = CholeskyVariationalDistribution(inducing_x.size(0))
        variational_strategy = VariationalStrategy(self, inducing_x, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(has_lengthscale=True, ard_num_dims=batch_x.shape[-1] if use_ard else None))

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
    return InducedKernelGPR(
        batch_x=torch.from_numpy(descriptors),
        inducing_x=torch.from_numpy(inducing_points),
        use_ard=use_ard_kernel,
    )


def train_model(model: InducedKernelGPR,
                train_x: np.ndarray,
                train_y: np.ndarray,
                steps: int,
                batch_size: int = 4,
                learning_rate: float = 0.01,
                verbose: bool = False) -> pd.DataFrame:
    """Train the kernel model over many iterations

    Args:
        model: Model to be trained
        train_x: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        train_y: Energies for each training point
        steps: Number of interactions over all training points
        batch_size: Number of conformers per batch
        learning_rate: Learning rate used for the optimizer
        verbose: Whether to display a progress bar
    Returns:
        Mean loss over each iteration
    """

    # Convert the data to Torches
    n_conf, n_atoms = train_x.shape[:2]
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    # Make the data loader
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Convert the model to training model
    likelihood = GaussianLikelihood()
    likelihood.train()
    model.train()

    # Define the optimizer and loss function
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = VariationalELBO(likelihood, model, num_data=train_y.shape[0])

    # Iterate over the data multiple times
    losses = []
    noises = []
    for _ in tqdm(range(steps), disable=not verbose, leave=False):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            # Prepare the beginning of each batch
            opt.zero_grad()

            # Predict on all configurations
            batch_x = torch.reshape(batch_x, (-1, batch_x.shape[-1]))  # Flatten from (n_confs, n_atoms, n_desc) -> (n_confs * n_atoms, n_desc)
            pred_y_per_atoms_flat = model(batch_x)

            # Get the mean sum for each atom
            pred_y_per_atoms = torch.reshape(pred_y_per_atoms_flat.mean, (batch_size, n_atoms))
            pred_y_mean = torch.sum(pred_y_per_atoms, dim=1)

            # The covariance matrix of those sums, assuming they are uncorrelated with each other (they are not)
            pred_y_covar_flat = pred_y_per_atoms_flat.covariance_matrix
            pred_y_covar_grouped_by_conf = pred_y_covar_flat.reshape((batch_size, n_atoms, batch_size, n_atoms))
            pred_y_covar = torch.sum(pred_y_covar_grouped_by_conf, dim=(1, 3))
            pred_y_covar = torch.diag(torch.diag(pred_y_covar))  # Make it diagonal

            # Turn them in to a distribution, and use that to compute a loss function
            pred_y_dist = MultivariateNormal(mean=pred_y_mean, covariance_matrix=pred_y_covar)

            # Compute loss and optimize
            loss = -mll(pred_y_dist, batch_y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss)
        noises.append(likelihood.noise.item())
    return pd.DataFrame({'loss': losses, 'noise': noises})


class SOAPCalculator(Calculator):
    """Calculator which uses a GPR model trained using SOAP descriptors

    Keyword Args:
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
        desc = torch.from_numpy(desc.astype(np.float32))
        desc.requires_grad = True
        d_desc_d_pos = torch.from_numpy(d_desc_d_pos.astype(np.float32))

        # Run inference
        offset, scale = self.parameters['scaling']
        model: InducedKernelGPR = self.parameters['model']
        model.eval()  # Ensure we're in eval mode
        pred_energies_dist = model(desc)
        pred_energies = pred_energies_dist.mean * scale + offset
        pred_energy = torch.sum(pred_energies)
        self.results['energy'] = pred_energy.detach().numpy()
        self.results['energies'] = pred_energy.detach().numpy()

        if 'forces' in properties:
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
