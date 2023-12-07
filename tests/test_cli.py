from contextlib import redirect_stdout
from os import devnull
from pathlib import Path

from jitterbug.cli import main


def test_exact_solver(xyz_path):
    with open(devnull, 'w') as fo:
        with redirect_stdout(fo):
            main([
                str(xyz_path), '--exact', '--method', 'pm7', 'None'
            ])
    assert (Path('run') / 'water' / 'pm7_none_exact' / 'hessian.npy').exists()


def test_parsl_path(xyz_path, file_dir):
    with open(devnull, 'w') as fo:
        with redirect_stdout(fo):
            main([
                str(xyz_path), '--exact', '--method', 'pm7', 'None',
                '--parsl-config', str(file_dir / 'example_config.py')
            ])
    assert (Path('run') / 'water' / 'pm7_none_exact' / 'hessian.npy').exists()
