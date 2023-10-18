import numpy as np
import torch
from dscribe.descriptors.soap import SOAP
from pytest import mark, fixture

from jitterbug.model.dscribe.local import make_gpr_model, train_model, DScribeLocalCalculator, DScribeLocalEnergyModel


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
    return soap.create(train_set)


@mark.parametrize('use_adr', [True, False])
def test_make_model(use_adr, descriptors, train_set):
    model = make_gpr_model(descriptors, 4, use_ard_kernel=use_adr)

    # Evaluate on a single point
    model.eval()
    pred_y = model(torch.from_numpy(descriptors[0, :, :]))
    assert pred_y.shape == (3,)  # 3 Atoms


@mark.parametrize('use_adr', [True, False])
def test_train(descriptors, train_set, use_adr):
    # Make the model and the training set
    train_y = np.array([a.get_potential_energy() for a in train_set])
    train_y -= train_y.min()
    model = make_gpr_model(descriptors, 4, use_ard_kernel=use_adr)
    model.inducing_x.requires_grad = False

    # Evaluate the untrained model
    model.eval()
    pred_y = model(torch.from_numpy(descriptors.reshape((-1, descriptors.shape[-1]))))
    assert pred_y.dtype == torch.float64
    error_y = pred_y.sum(axis=-1).detach().numpy() - train_y
    mae_untrained = np.abs(error_y).mean()

    # Train
    losses = train_model(model, descriptors, train_y, 64)
    assert len(losses) == 64

    # Run the evaluation
    model.eval()
    pred_y = model(torch.from_numpy(descriptors.reshape((-1, descriptors.shape[-1]))))
    error_y = pred_y.sum(axis=-1).detach().numpy() - train_y
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
    train_model(model, descriptors, train_y, 32)

    # Make the model
    calc = DScribeLocalCalculator(
        model=model,
        desc=soap,
        desc_scaling=(offset_x, scale_x),
    )
    energies = []
    for atoms in train_set:
        atoms.calc = calc
        forces = atoms.get_forces()
        energies.append(atoms.get_potential_energy())
        numerical_forces = calc.calculate_numerical_forces(atoms, d=1e-4)
        assert np.isclose(forces[:, :2], numerical_forces[:, :2], rtol=5e-1).all()  # Make them agree w/i 50% (PES is not smooth)
    assert np.std(energies) > 1e-6


def test_model(soap, train_set):
    # Assemble the model
    model = DScribeLocalEnergyModel(
        reference=train_set[0],
        descriptors=soap,
        model_fn=lambda x: make_gpr_model(x, num_inducing_points=32),
        num_calculators=4,
    )

    # Run the fitting
    calcs = model.train(train_set)

    # Test the mean hessian function
    mean_hess = model.mean_hessian(calcs)
    assert mean_hess.shape == (9, 9), 'Wrong shape'
    assert np.isclose(mean_hess, mean_hess.T).all(), 'Not symmetric'

    # Test the sampling
    sampled_hess = model.sample_hessians(calcs, 128)
    assert all(np.isclose(hess, hess.T).all() for hess in sampled_hess)
    mean_sampled_hess = np.mean(sampled_hess, 0)
    assert np.isclose(np.diag(mean_sampled_hess), np.diag(mean_hess), atol=5).mean() > 0.5  # Make sure most agree
