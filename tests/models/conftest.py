from pathlib import Path

from ase import Atoms
from ase.db import connect
from pytest import fixture

file_dir = Path(__file__).parent / 'files'


@fixture()
def train_set() -> list[Atoms]:
    with connect(file_dir / 'water_hf_def2-svpd-random-d=2.00e-02.db') as db:
        return [a.toatoms() for a in db.select('')]
