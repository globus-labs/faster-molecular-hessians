import numpy as np
import torch
from dscribe.descriptors.soap import SOAP
from pytest import mark, fixture

from jitterbug.model.soap import make_gpr_model, train_model, SOAPCalculator


@fixture
def soap(train_set):
    species = sorted(set(sum([a.get_chemical_symbols() for a in train_set], [])))
    return SOAP(
        species=species,
        r_cut=4.,
        n_max=4,
        l_max=4,
    )


@fixture
def descriptors(train_set, soap):
    return soap.create(train_set).astype(np.float32)


@mark.parametrize('use_adr', [True, False])
def test_make_model(use_adr, descriptors, train_set):
    model = make_gpr_model(descriptors, 4, use_ard_kernel=use_adr)

    # Evaluate on a single point
    model.eval()
    pred_dist = model(torch.from_numpy(descriptors[0, :, :]))
    assert torch.isclose(pred_dist.mean, torch.zeros((1, 3,)), atol=1e-2).all(), [pred_dist.mean]


def test_train(descriptors, train_set):
    # Make the model and the training set
    train_y = np.array([a.get_potential_energy() for a in train_set])
    train_y -= train_y.min()
    model = make_gpr_model(descriptors, 4)

    # Evaluate the untrained model
    model.eval()
    pred_y = model(torch.from_numpy(descriptors))
    error_y = pred_y.mean.sum(dim=1).detach().numpy() - train_y
    mae_untrained = np.abs(error_y).mean()

    # Train
    losses = train_model(model, descriptors, train_y, 64)
    assert len(losses) == 64

    # Run the evaluation
    model.eval()
    pred_y = model(torch.from_numpy(descriptors))
    error_y = pred_y.mean.sum(dim=1).detach().numpy() - train_y
    mae_trained = np.abs(error_y).mean()
    assert mae_trained < mae_untrained


def test_calculator(descriptors, soap, train_set):
    # Scale the input and outputs
    train_y = np.array([a.get_potential_energy() for a in train_set])
    train_y -= train_y.mean()

    offset_x = descriptors.mean(axis=(0, 1))
    scale_x = np.clip(descriptors.std(axis=(0, 1)), a_min=1e-6, a_max=None)
    descriptors = (descriptors - offset_x) / scale_x

    # Assemble and train for a few instances so that we get nonzero forces
    model = make_gpr_model(descriptors, 32)
    train_model(model, descriptors, train_y, 16)

    # Make the model
    calc = SOAPCalculator(
        model=model,
        soap=soap,
        desc_scaling=(offset_x, scale_x),
    )
    energies = []
    for atoms in train_set:
        atoms.calc = calc
        forces = atoms.get_forces()
        energies.append(atoms.get_potential_energy())
        numerical_forces = calc.calculate_numerical_forces(atoms)
        assert np.isclose(forces, numerical_forces, atol=1e-1).all()
    assert np.std(energies) > 1e-6
