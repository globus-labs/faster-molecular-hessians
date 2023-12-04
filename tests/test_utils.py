from pytest import raises

from jitterbug.utils import make_calculator


def test_make_calculator():
    assert make_calculator('b3lyp', 'def2_svpd').name == 'psi4'
    assert make_calculator('pm7', None).name == 'mopac'

    with raises(ValueError):
        make_calculator('pm7', 'wrong')
