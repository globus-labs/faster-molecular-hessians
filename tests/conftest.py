from pathlib import Path

from pytest import fixture

_file_dir = Path(__file__).parent / 'files'


@fixture()
def file_dir():
    return _file_dir


@fixture()
def xyz_path():
    return _file_dir / 'water.xyz'
