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
    assert (Path('run') / 'water' / 'pm7_none' / 'hessian.npy').exists()
