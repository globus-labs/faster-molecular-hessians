from pathlib import Path

from ase.vibrations import VibrationsData
from pytest import fixture
import numpy as np

from jitterbug.compare import compare_hessians

_test_files = Path(__file__).parent / 'files'


@fixture()
def example_hess() -> VibrationsData:
    return VibrationsData.read(_test_files / 'water-hessian.json')


def test_compare(example_hess):
    comp = compare_hessians(example_hess.get_atoms(), example_hess.get_hessian_2d(), example_hess.get_hessian_2d())
    assert comp.zpe_error == 0.
    assert np.ndim(comp.cp_error) == 1
    assert np.mean(comp.cp_error) == 0.
