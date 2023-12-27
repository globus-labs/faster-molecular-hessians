import numpy as np
import torch
from dscribe.descriptors.soap import SOAP
from pytest import fixture

from jitterbug.model.dscribe.local import make_gpr_model, train_model, DScribeLocalCalculator, DScribeLocalEnergyModel, make_nn_model


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


@fixture
def elements(train_set):
    return train_set[0].get_atomic_numbers()


@fixture(params=['gpr-ard', 'gpr', 'nn'])
def model(elements, descriptors, request):
    if request.param == 'gpr':
        return make_gpr_model(elements, descriptors, 4, use_ard_kernel=False)
    elif request.param == 'gpr-ard':
        return make_gpr_model(elements, descriptors, 4, use_ard_kernel=True)
    elif request.param == 'nn':
        return make_nn_model(elements, descriptors, (16, 16))
    else:
        raise NotImplementedError()


def test_make_model(model, elements, descriptors, train_set):
    # Evaluate on a single point
    model.eval()
    pred_y = model(
        torch.from_numpy(elements),
        torch.from_numpy(descriptors[0, :, :])
    )
    assert pred_y.shape == (3,)  # 3 Atoms


def test_train(model, elements, descriptors, train_set):
    # Make the model and the training set
    train_y = np.array([a.get_potential_energy() for a in train_set])
    train_y -= train_y.min()

    # Evaluate the untrained model
    model.eval()
    pred_y = model(
        torch.from_numpy(np.repeat(elements, descriptors.shape[0])),
        torch.from_numpy(descriptors.reshape((-1, descriptors.shape[-1])))
    )
    assert pred_y.dtype == torch.float64
    pred_y = torch.reshape(pred_y, [-1, elements.shape[0]])
    error_y = pred_y.sum(axis=-1).detach().numpy() - train_y
    mae_untrained = np.abs(error_y).mean()

    # Train
    losses = train_model(model, elements, descriptors, train_y, learning_rate=0.001, steps=8)
    assert len(losses) == 8

    # Run the evaluation
    model.eval()
    pred_y = model(
        torch.from_numpy(np.repeat(elements, descriptors.shape[0])),
        torch.from_numpy(descriptors.reshape((-1, descriptors.shape[-1])))
    )
    pred_y = torch.reshape(pred_y, [-1, elements.shape[0]])
    error_y = pred_y.sum(axis=-1).detach().numpy() - train_y
    mae_trained = np.abs(error_y).mean()
    assert mae_trained < mae_untrained * 1.1


def test_calculator(elements, descriptors, soap, train_set):
    # Scale the input and outputs
    train_y = np.array([a.get_potential_energy() for a in train_set])
    train_y -= train_y.mean()

    offset_x = descriptors.mean(axis=(0, 1))
    scale_x = np.clip(descriptors.std(axis=(0, 1)), a_min=1e-6, a_max=None)
    descriptors = (descriptors - offset_x) / scale_x

    # Assemble and train for a few instances so that we get nonzero forces
    model = make_gpr_model(elements, descriptors, 32)
    train_model(model, elements, descriptors, train_y, 32)

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
        force_mask = np.abs(numerical_forces) > 0.1
        assert np.isclose(forces[force_mask], numerical_forces[force_mask], rtol=0.1).all()  # Agree w/i 10%
    assert np.std(energies) > 1e-6


def test_model(soap, train_set):
    # Assemble the model
    model = DScribeLocalEnergyModel(
        reference=train_set[0],
        descriptors=soap,
        model_fn=lambda x: make_gpr_model(train_set[0].get_atomic_numbers(), x, num_inducing_points=32),
        num_calculators=4,
    )

    # Run the fitting
    calcs = model.train(train_set)

    # Make sure the energy is reasonable
    eng = calcs[0].get_potential_energy(train_set[0])
    assert np.isclose(eng, train_set[0].get_potential_energy(), atol=1e-2)

    # Make sure they differ between entries
    pred_e = [calcs[0].get_potential_energy(a) for a in train_set]
    assert np.std(pred_e) > 1e-3

    # Test the mean hessian function
    mean_hess = model.mean_hessian(calcs)
    assert mean_hess.shape == (9, 9), 'Wrong shape'
    assert np.isclose(mean_hess, mean_hess.T).all(), 'Not symmetric'

    # Test the sampling
    sampled_hess = model.sample_hessians(calcs, 128)
    assert all(np.isclose(hess, hess.T).all() for hess in sampled_hess)
    mean_sampled_hess = np.mean(sampled_hess, 0)
    assert np.isclose(np.diag(mean_sampled_hess), np.diag(mean_hess), atol=5).mean() > 0.5  # Make sure most agree
