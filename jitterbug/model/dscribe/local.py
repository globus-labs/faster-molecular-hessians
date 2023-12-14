"""Create a PyTorch-based model which uses features for each atom"""
from typing import Union, Optional, Callable

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
    """

    def __init__(self, inducing_x: torch.Tensor, use_ard: bool):
        super().__init__()
        n_points, n_desc = inducing_x.shape
        self.inducing_x = torch.nn.Parameter(inducing_x.clone())
        self.alpha = torch.nn.Parameter(torch.empty((n_points,), dtype=inducing_x.dtype))
        torch.nn.init.normal_(self.alpha)
        self.lengthscales = torch.nn.Parameter(-torch.ones((n_desc,), dtype=inducing_x.dtype) if use_ard else -torch.ones((1,), dtype=inducing_x.dtype))

    def forward(self, x) -> torch.Tensor:
        # Compute an RBF kernel
        lengthscales = torch.exp(self.lengthscales)
        diff_sq = torch.pow(x[None, :, :] - self.inducing_x[:, None, :], 2) / lengthscales
        diff = diff_sq.sum(axis=-1)  # Sum along the descriptor axis
        esd = torch.exp(-diff)

        # Return the sum
        return torch.tensordot(self.alpha, esd, dims=([0], [0]))


class PerElementModel(torch.nn.Module):
    """Train a different machine learning model for each element"""

    def __init__(self, models: list[InducedKernelGPR], dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.models = torch.nn.ModuleList(models)

    def forward(self, element: torch.IntTensor, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(x.shape[0], dtype=self.dtype).to(element.device)

        for i, model in enumerate(self.models):
            mask = element == i
            energies = model(x[mask, :])
            output[mask] = energies

        return output


def make_gpr_model(
        num_elements: int,
        train_descriptors: np.ndarray,
        num_inducing_points: int,
        use_ard_kernel: bool = False
) -> PerElementModel:
    """Make the GPR model for a certain atomic system

    Assumes that the descriptors have already been scaled

    Args:
        num_elements: Number of elements to include in model
        train_descriptors: 3D array of all training points (num_configurations, num_atoms, num_descriptors)
        num_inducing_points: Number of inducing points to use in the kernel. More points, more complex model
        use_ard_kernel: Whether to use a different length scale parameter for each descriptor
    Returns:
        Model which can predict energy given descriptors for a single configuration
    """

    # Select a set of inducing points randomly
    descriptors = np.reshape(train_descriptors, (-1, train_descriptors.shape[-1]))
    num_inducing_points = min(num_inducing_points, descriptors.shape[0])
    inducing_inds = np.random.choice(descriptors.shape[0], size=(num_inducing_points,), replace=False)
    inducing_points = descriptors[inducing_inds, :]

    # Make a model for each element
    models = [
        InducedKernelGPR(
            inducing_x=torch.from_numpy(inducing_points),
            use_ard=use_ard_kernel,
        ) for _ in range(num_elements)
    ]

    # Combine to form a big model
    return PerElementModel(models)


def train_model(model: torch.nn.Module,
                train_elem: np.ndarray,
                train_desc: np.ndarray,
                train_y: np.ndarray,
                steps: int,
                batch_size: int = 4,
                learning_rate: float = 0.01,
                device: Union[str, torch.device] = 'cpu',
                verbose: bool = False) -> pd.DataFrame:
    """Train the kernel model over many iterations

    Assumes that the descriptors and energies have already been scaled

    Args:
        model: Model to be trained
        train_elem: 2D array of atomic numbers (num_configurations, num_atoms)
        train_desc: 3D array of the descriptors all training points (num_configurations, num_atoms, num_descriptors)
        train_y: Energies for each training point
        steps: Number of interactions over all training points
        batch_size: Number of conformers per batch
        learning_rate: Learning rate used for the optimizer
        device: Which device to use for training
        verbose: Whether to display a progress bar
    Returns:
        Mean loss over each iteration
    """

    # Convert element types to indices
    elem_types = np.unique(train_elem)
    elem_ind_x = np.empty_like(train_elem)
    for i, z in enumerate(elem_types):
        elem_ind_x[train_elem == z] = i

    # Convert the data to Torches
    n_conf, n_atoms = train_desc.shape[:2]
    train_desc = torch.from_numpy(train_desc)
    train_y = torch.from_numpy(train_y)
    elem_ind_x = torch.from_numpy(elem_ind_x)

    # Make the data loader
    dataset = TensorDataset(elem_ind_x, train_desc, train_y)
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
    for _ in tqdm(range(steps), disable=not verbose, leave=False):
        epoch_loss = 0
        for batch_elem, batch_desc, batch_y in loader:
            # Prepare at the beginning of each batch
            opt.zero_grad()
            batch_elem = batch_elem.to(device)
            batch_desc = batch_desc.to(device)
            batch_y = batch_y.to(device)

            # Predict on all configurations
            batch_elem = torch.reshape(batch_elem, (-1,))
            batch_desc = torch.reshape(batch_desc, (-1, batch_desc.shape[-1]))  # Flatten from (n_confs, n_atoms, n_desc) -> (n_confs * n_atoms, n_desc)
            pred_y_per_atoms_flat = model(batch_elem, batch_desc)

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

        losses.append(epoch_loss)

    # Pull the model back off the GPU
    model.to('cpu')
    return pd.DataFrame({'loss': losses})


class DScribeLocalCalculator(Calculator):
    """Calculator which uses descriptors for each atom and PyTorch to compute energy

    Keyword Args:
        element_list (list[int]): Atomic numbers of elements used in model
        model (torch.nn.Module): A machine learning model which takes descriptors as inputs
        desc (DescriptorLocal): Tool used to compute the descriptors
        desc_scaling (tuple[np.ndarray, np.ndarray]): A offset and factor with which to adjust the energy per atom predictions,
            which are typically he mean and standard deviation of energy per atom across the training set.
        energy_scaling (tuple[float, float]): A offset and factor with which to adjust the energy per atom predictions,
            which are typically he mean and standard deviation of energy per atom across the training set.
        device (str | torch.device): Device to use for inference
    """

    implemented_properties = ['energy', 'forces', 'energies']
    default_parameters = {
        'model': None,
        'desc': None,
        'desc_scaling': (0., 1.),
        'energy_scaling': (0., 1.),
        'element_list': (),
        'device': 'cpu'
    }

    def calculate(self, atoms=None, properties=('energy', 'forces', 'energies'),
                  system_changes=all_changes):
        # Get the atoms
        elem_list = self.parameters['element_list']
        if any(x not in elem_list for x in atoms.get_atomic_numbers()):
            raise ValueError('Atoms object contains elements not included in the potential')
        elems = np.array([elem_list.index(z) for z in atoms.get_atomic_numbers()])

        # Compute the descriptors for the atoms
        d_desc_d_pos, desc = self.parameters['desc'].derivatives(atoms, attach=True)

        # Scale the descriptors
        offset, scale = self.parameters['desc_scaling']
        desc = (desc - offset) / scale
        d_desc_d_pos /= scale

        # Convert to pytorch
        desc = torch.from_numpy(desc.astype(np.float32))
        desc.requires_grad = True
        d_desc_d_pos = torch.from_numpy(d_desc_d_pos.astype(np.float32))
        elems = torch.from_numpy(elems)

        # Move the model to device if need be
        model: torch.nn.Module = self.parameters['model']
        device = self.parameters['device']
        model.to(device)

        # Run inference
        offset, scale = self.parameters['energy_scaling']
        model.eval()  # Ensure we're in eval mode
        elems = elems.to(device)
        desc = desc.to(device)
        pred_energies_dist = model(elems, desc)
        pred_energies = pred_energies_dist * scale + offset
        pred_energy = torch.sum(pred_energies)
        self.results['energy'] = pred_energy.item()
        self.results['energies'] = pred_energies.detach().cpu().numpy()

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
            d_desc_d_pos = d_desc_d_pos.to(device)
            d_energy_d_center_d_pos = torch.einsum('ijkl,il->ijk', d_desc_d_pos, d_energy_d_desc)  # Derivative for each center with respect to each atom
            pred_forces = -d_energy_d_center_d_pos.sum(dim=0) * scale  # Total effect on each center from each atom

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
                 model_fn: Callable[[np.ndarray], torch.nn.Module],
                 num_calculators: int,
                 device: str = 'cpu',
                 train_options: Optional[dict] = None):
        super().__init__(reference, num_calculators)
        self.descriptors = descriptors
        self.model_fn = model_fn
        self.device = device
        self.train_options = train_options or {'steps': 4}

    def train_calculator(self, data: list[Atoms]) -> Calculator:
        # Train it using the user-provided options
        train_x = self.descriptors.create(data).astype(np.float32)
        offset_x = train_x.mean(axis=(0, 1))
        scale_x = np.clip(train_x.std(axis=(0, 1)), a_min=1e-6, a_max=None)
        train_x -= offset_x
        train_x /= scale_x

        train_y = np.array([a.get_potential_energy() for a in data]).astype(np.float32)
        scale_y, offset_y = np.std(train_y), np.mean(train_y)
        train_y = (train_y - offset_y) / scale_y

        # Get the element for each atom
        elements = np.array([x.get_atomic_numbers() for x in data])

        # Make then train the model
        model = self.model_fn(train_x)
        train_model(model, elements, train_x, train_y, device=self.device, **self.train_options)

        # Make the calculator
        return DScribeLocalCalculator(
            element_list=np.unique(elements).tolist(),
            model=model,
            desc=self.descriptors,
            desc_scaling=(offset_x, scale_x),
            energy_scaling=(offset_y, scale_y),
            device=self.device
        )
