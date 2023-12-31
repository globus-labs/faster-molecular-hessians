"""Create a PyTorch-based model which uses features for each atom"""
from typing import Union, Optional, Callable, Sequence

import ase
from ase.calculators.calculator import Calculator, all_changes
from dscribe.descriptors.descriptorlocal import DescriptorLocal
from torch.utils.data import TensorDataset, DataLoader
from ase import Atoms
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from jitterbug.model.base import ASEEnergyModel


class InducedKernelGPR(torch.nn.Module):
    """Gaussian process regression model with an induced kernel

    Predicts the energy for each atom as a function of its descriptors

    Args:
        inducing_x: Starting points for the reference points of the kernel
        use_ard: Whether to employ a different length scale parameter for each descriptor,
            a technique known as Automatic Relevance Detection (ARD)
        initial_lengthscale: Initial value for the lengthscale parmaeter
    """

    def __init__(self, inducing_x: torch.Tensor, use_ard: bool, initial_lengthscale: float = 10):
        super().__init__()
        n_points, n_desc = inducing_x.shape
        self.inducing_x = torch.nn.Parameter(inducing_x.clone())
        self.alpha = torch.nn.Parameter(torch.empty((n_points,), dtype=inducing_x.dtype))
        torch.nn.init.normal_(self.alpha)

        # Initial value
        ls = np.log(initial_lengthscale)
        self.lengthscales = torch.nn.Parameter(
            -torch.ones((n_desc,), dtype=inducing_x.dtype) * ls if use_ard else
            -torch.ones((1,), dtype=inducing_x.dtype) * ls)

    def forward(self, x) -> torch.Tensor:
        # Compute an RBF kernel
        lengthscales = torch.exp(self.lengthscales)
        diff_sq = torch.pow(x[None, :, :] - self.inducing_x[:, None, :], 2) / lengthscales
        diff = diff_sq.sum(axis=-1)  # Sum along the descriptor axis
        esd = torch.exp(-diff)

        # Return the sum
        return torch.tensordot(self.alpha, esd, dims=([0], [0]))[:, None]


class PerElementModule(torch.nn.Module):
    """Fit a different model for each element

    Args:
        models: Map of atomic number to element to use
    """

    def __init__(self, models: dict[int, torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleDict(
            dict((str(k), v) for k, v in models.items())
        )

    def forward(self, element: torch.IntTensor, desc: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(desc[:, 0])
        for elem, model in self.models.items():
            elem_id = int(elem)
            mask = element == elem_id
            output[mask] = model(desc[mask, :])[:, 0]
        return output


def make_nn_model(
        elements: np.ndarray,
        descriptors: np.ndarray,
        hidden_layers: Sequence[int] = (),
        activation: torch.nn.Module = torch.nn.Sigmoid()
) -> PerElementModule:
    """Make a neural network model for a certain atomic system

    Assumes that the descriptors have already been scaled

    Args:
        elements: Element for each atom in a structure (num_atoms,)
        descriptors: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        hidden_layers: Number of units in the hidden layers
        activation: Activation function used for the hidden layers
    """

    # Detect the dtype
    dtype = torch.from_numpy(descriptors[0, 0, :1]).dtype

    # Make a model for each element type
    models: dict[int, torch.nn.Sequential] = {}
    element_types = np.unique(elements)

    for element in element_types:
        # Make the neural network
        nn_layers = []
        input_size = descriptors.shape[2]
        for hidden_size in hidden_layers:
            nn_layers.extend([
                torch.nn.Linear(input_size, hidden_size, dtype=dtype),
                activation
            ])
            input_size = hidden_size

        # Make the last layer
        nn_layers.append(torch.nn.Linear(input_size, 1, dtype=dtype))
        models[element] = torch.nn.Sequential(*nn_layers)

    return PerElementModule(models)


def make_gpr_model(elements: np.ndarray,
                   descriptors: np.ndarray,
                   num_inducing_points: int,
                   fix_inducing_points: bool = True,
                   use_ard_kernel: bool = False,
                   **kwargs) -> PerElementModule:
    """Make the GPR model for a certain atomic system

    Assumes that the descriptors have already been scaled.

    Passes additional kwargs to the :class:`InducedKernelGPR` constructor.

    Args:
        elements: Element for each atom in a structure (num_atoms,)
        descriptors: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        num_inducing_points: Number of inducing points to use in the kernel for each model. More points, more complex model
        fix_inducing_points: Whether to fix the inducing points or allow them to be learned
        use_ard_kernel: Whether to use a different length scale parameter for each descriptor
    Returns:
        Model which can predict energy given descriptors for a single configuration
    """

    # Make a model for each element type
    models: dict[int, InducedKernelGPR] = {}
    element_types = np.unique(elements)

    for element in element_types:
        # Select a set of inducing points from records of each atom
        #  TODO (wardlt): Use a method which ensures diversity, like KMeans
        mask = elements == element
        masked_descriptors = descriptors[:, mask, :]
        masked_descriptors = np.reshape(masked_descriptors, (-1, masked_descriptors.shape[-1]))
        num_inducing_points = min(num_inducing_points, masked_descriptors.shape[0])
        inducing_inds = np.random.choice(masked_descriptors.shape[0], size=(num_inducing_points,), replace=False)
        inducing_points = masked_descriptors[inducing_inds, :]

        # Make the model
        model = InducedKernelGPR(
            inducing_x=torch.from_numpy(inducing_points),
            use_ard=use_ard_kernel,
            **kwargs
        )
        model.inducing_x.requires_grad = not fix_inducing_points
        models[element] = model

    return PerElementModule(models)


def train_model(model: PerElementModule,
                train_e: np.ndarray,
                train_x: np.ndarray,
                train_y: np.ndarray,
                steps: int,
                batch_size: int = 4,
                learning_rate: float = 0.01,
                device: Union[str, torch.device] = 'cpu',
                patience: Optional[int] = None,
                verbose: bool = False) -> pd.DataFrame:
    """Train the kernel model over many iterations

    Assumes that the descriptors have already been scaled

    Args:
        model: Model to be trained
        train_e: Elements for each atom in a configuration (num_atoms,)
        train_x: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        train_y: Energies for each training point
        steps: Number of interactions over all training points
        batch_size: Number of conformers per batch
        learning_rate: Learning rate used for the optimizer
        device: Which device to use for training
        patience: If provided, stop learning if train loss fails to improve after these many iterations
        verbose: Whether to display a progress bar
    Returns:
        Mean loss over each iteration
    """

    # Convert the data to Tensors
    n_conf, n_atoms = train_x.shape[:2]
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    train_e = torch.from_numpy(train_e).to(device)

    # Duplicate the elements per batch size
    train_e = train_e.repeat(batch_size)

    # Make the data loader
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Convert the model to training mode on the device
    model.train()
    model.to(device)

    # Define the optimizer and loss function
    opt = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=learning_rate)
    loss = torch.nn.MSELoss()

    # Iterate over the data multiple times
    losses = []
    iterator = tqdm(range(steps), disable=not verbose, leave=False)
    no_improvement = 0  # Number of epochs w/o improvement
    best_loss = np.inf
    for _ in iterator:
        epoch_loss = 0
        for batch_x, batch_y in loader:
            # Prepare at the beginning of each batch
            opt.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Predict on all configurations
            batch_x = torch.reshape(batch_x, (-1, batch_x.shape[-1]))  # Flatten from (n_confs, n_atoms, n_desc) -> (n_confs * n_atoms, n_desc)
            pred_y_per_atoms_flat = model(train_e, batch_x)

            # Get the mean sum for each atom
            pred_y_per_atoms = torch.reshape(pred_y_per_atoms_flat, (batch_size, n_atoms))
            pred_y = torch.sum(pred_y_per_atoms, dim=1)

            # Compute loss and propagate
            batch_loss = loss(pred_y, batch_y)
            if torch.isnan(batch_loss):
                raise ValueError('NaN loss')
            batch_loss.backward()

            opt.step()
            epoch_loss += batch_loss.item()

        # Update the best loss
        no_improvement = 0 if epoch_loss < best_loss else no_improvement + 1
        best_loss = min(best_loss, epoch_loss)
        iterator.set_description(f'Loss: {epoch_loss:.2e} - Patience: {no_improvement}')
        losses.append(epoch_loss)

        # Break if no improvement
        if patience is not None and no_improvement > patience:
            break

    # Pull the model back off the GPU
    model.to('cpu')
    return pd.DataFrame({'loss': losses})


class DScribeLocalCalculator(Calculator):
    """Calculator which uses descriptors for each atom and PyTorch to compute energy

    Keyword Args:
        model (PerElementModule): A machine learning model which takes atomic numbers and descriptors as inputs
        desc (DescriptorLocal): Tool used to compute the descriptors
        desc_scaling (tuple[np.ndarray, np.ndarray]): A offset and factor with which to adjust the energy per atom predictions,
            which are typically he mean and standard deviation of energy per atom across the training set.
        energy_scaling (tuple[float, float]): A offset and factor with which to adjust the energy per atom predictions,
            which are typically he mean and standard deviation of energy per atom across the training set.
        device (str | torch.device): Device to use for inference
    """

    # TODO (wardlt): Have scaling for descriptors and energies be per-element

    implemented_properties = ['energy', 'forces', 'energies']
    default_parameters = {
        'model': None,
        'desc': None,
        'desc_scaling': (0., 1.),
        'energy_scaling': (0., 1.),
        'device': 'cpu'
    }

    def calculate(self, atoms: ase.Atoms = None, properties=('energy', 'forces', 'energies'),
                  system_changes=all_changes):
        # Compute the descriptors for the atoms
        d_desc_d_pos, desc = self.parameters['desc'].derivatives(atoms, attach=True)

        # Scale the descriptors
        desc_offset, desc_scale = self.parameters['desc_scaling']
        desc = (desc - desc_offset) / desc_scale
        d_desc_d_pos /= desc_scale

        # Convert to pytorch
        #  TODO (wardlt): Make it possible to convert to float32 or lower
        desc = torch.from_numpy(desc)
        desc.requires_grad = True
        d_desc_d_pos = torch.from_numpy(d_desc_d_pos)

        # Make sure the model has all the required elements
        model: PerElementModule = self.parameters['model']
        missing_elems = set(map(str, atoms.get_atomic_numbers())).difference(model.models.keys())
        if len(missing_elems) > 0:
            raise ValueError(f'Model lacks parameters for elements: {", ".join(missing_elems)}')
        device = self.parameters['device']
        model.to(device)

        # Run inference
        eng_offset, eng_scale = self.parameters['energy_scaling']
        elements = torch.from_numpy(atoms.get_atomic_numbers())
        model.eval()  # Ensure we're in eval mode
        elements = elements.to(device)
        desc = desc.to(device)
        pred_energies_dist = model(elements, desc)
        pred_energies = pred_energies_dist * eng_scale + eng_offset
        pred_energy = torch.sum(pred_energies)
        self.results['energy'] = pred_energy.item()
        self.results['energies'] = pred_energies.detach().cpu().numpy()

        if 'forces' in properties:
            # Compute the forces
            #  See: https://singroup.github.io/dscribe/latest/tutorials/machine_learning/forces_and_energies.html
            # Derivatives for the descriptors are for each center (which is the input to the model) with respect to each atomic coordinate changing.
            # Energy is summed over the contributions from each center.
            # The total force is therefore a sum over the effect of an atom moving on all centers
            # Note: Forces are scaled because pred_energy was scaled
            d_energy_d_desc = torch.autograd.grad(
                outputs=pred_energy,
                inputs=desc,
                grad_outputs=torch.ones_like(pred_energy),
            )[0]  # Derivative of the energy with respect to the descriptors for each center
            d_desc_d_pos = d_desc_d_pos.to(device)

            # Einsum is log-form for: dE_d(center:i from atom:j moving direction:k)
            #  = sum_(descriptors:l) d(descriptor:l)/d(i,j,k) * dE(center:i)/d(l)
            # "Use the chain rule to get the change in energy for each center
            d_energy_d_center_d_pos = torch.einsum('ijkl,il->ijk', d_desc_d_pos, d_energy_d_desc)
            pred_forces = -d_energy_d_center_d_pos.sum(dim=0)  # Total effect on each center from each atom

            # Store the results
            self.results['forces'] = pred_forces.detach().cpu().numpy()

        # Move the model back to CPU memory
        model.to('cpu')


class DScribeLocalEnergyModel(ASEEnergyModel):
    """Energy model based on DScribe atom-level descriptors

    Trains an energy model using PyTorch

    Args:
        reference: Reference structure at which we compute the Hessian
        descriptors: Tool used to compute descriptors
        model_fn: Function used to create the model given descriptors for the training set
        num_calculators: Number of models to use in ensemble
        device: Device used for training
        train_options: Options passed to the training function
    """

    def __init__(self,
                 reference: Atoms,
                 descriptors: DescriptorLocal,
                 model_fn: Callable[[np.ndarray], PerElementModule],
                 num_calculators: int,
                 device: str = 'cpu',
                 train_options: Optional[dict] = None):
        super().__init__(reference, num_calculators)
        self.descriptors = descriptors
        self.model_fn = model_fn
        self.device = device
        self.train_options = train_options or {'steps': 4}

    def train_calculator(self, data: list[Atoms]) -> Calculator:
        # Get the elements
        elements = data[0].get_atomic_numbers()

        # Prepare the training set, scaling the input
        train_x = self.descriptors.create(data)
        offset_x = train_x.mean(axis=(0, 1))
        train_x -= offset_x
        scale_x = np.clip(train_x.std(axis=(0, 1)), a_min=1e-6, a_max=None)
        train_x /= scale_x

        train_y_per_atom = np.array([a.get_potential_energy() / len(a) for a in data])
        scale, offset = train_y_per_atom.std(), train_y_per_atom.mean()
        train_y = np.array([(a.get_potential_energy() - len(a) * offset) / scale for a in data])

        # Make the model and train it
        model = self.model_fn(train_x)
        train_model(model, elements, train_x, train_y, device=self.device, **self.train_options)

        # Return the model
        return DScribeLocalCalculator(
            model=model,
            desc=self.descriptors,
            desc_scaling=(offset_x, scale_x),
            energy_scaling=(offset, scale),
            device=self.device
        )
