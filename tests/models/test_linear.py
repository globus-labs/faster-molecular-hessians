import numpy as np
from ase.build import molecule
from ase.vibrations import VibrationsData

from jitterbug.model.linear import get_displacement_matrix, LinearHessianModel


def test_disp_matrix():
    reference = molecule('H2')
    atoms = reference.copy()

    # With a single displacement only the first term should be nonzero
    atoms.positions[0, 0] += 0.1
    disp_matrix = get_displacement_matrix(atoms, reference)
    assert disp_matrix.size == 21
    assert (disp_matrix != 0).sum() == 1
    assert np.isclose(disp_matrix[0], 0.01)

    # With two displacements, there should be 3 nonzero terms
    atoms.positions[1, 0] += 0.05
    disp_matrix = get_displacement_matrix(atoms, reference)
    assert (disp_matrix != 0).sum() == 3
    assert np.isclose(disp_matrix[0], 0.01)  # (Atom 0, x) * (Atom 0, x)
    assert np.isclose(disp_matrix[3], 0.1 * 0.05 * 2)  # (Atom 0, x) * (Atom 1, x) * 2 (harmonic)
    assert np.isclose(disp_matrix[6 + 5 + 4], 0.0025)  # (Atom 1, x) * (Atom 1, x)


def test_linear_model(train_set):
    # The first atom in the set should have forces
    reference = train_set[0]
    assert reference.get_forces().max() < 0.01

    # Fit the model
    model = LinearHessianModel(reference)
    hessian_model = model.train(train_set)

    # Sample the Hessians, at least make sure the results are near correct
    hessians = model.sample_hessians(hessian_model, num_samples=32)
    assert len(hessians)
    assert np.isclose(hessians[0], hessians[0].T).all()

    # Create a vibration data object
    vib_data = VibrationsData.from_2d(reference, hessians[0])
    zpe = vib_data.get_zero_point_energy()
    print(zpe)
    assert np.isclose(zpe, 0.63, atol=0.3)  # Make sure it's _close_
