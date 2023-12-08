from contextlib import redirect_stdout
from os import devnull
from pathlib import Path

from pytest import mark

from jitterbug.cli import main


@mark.parametrize('xyz', ['water.xyz', 'radical.extxyz'])
def test_exact_solver(file_dir, xyz):
    xyz_name = Path(xyz).with_suffix('').name
    with open(devnull, 'w') as fo:
        with redirect_stdout(fo):
            main([
                str(file_dir / xyz), '--exact', '--method', 'hf', 'sto-3g'
            ])
    assert (Path('run') / xyz_name / 'hf_sto-3g_exact' / 'hessian.npy').exists()


def test_parsl_path(xyz_path, file_dir):
    with open(devnull, 'w') as fo:
        with redirect_stdout(fo):
            main([
                str(xyz_path), '--exact', '--method', 'pm7', 'None',
                '--parsl-config', str(file_dir / 'example_config.py')
            ])
    assert (Path('run') / 'water' / 'pm7_none_exact' / 'hessian.npy').exists()
