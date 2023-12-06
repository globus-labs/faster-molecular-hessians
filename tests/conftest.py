from pathlib import Path

from pytest import fixture

_file_dir = Path(__file__).parent / 'files'


@fixture()
def xyz_path():
    return _file_dir / 'water.xyz'
