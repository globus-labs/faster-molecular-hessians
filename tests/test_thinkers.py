from functools import partial, update_wrapper
from pathlib import Path

import numpy as np
from ase.io import read
from colmena.queue.python import PipeQueues
from colmena.task_server.parsl import ParslTaskServer
from parsl import Config, HighThroughputExecutor
from pytest import fixture

from jitterbug.parsl import get_energy
from jitterbug.thinkers.exact import ExactHessianThinker


@fixture()
def queues():
    return PipeQueues(topics='simulation')


@fixture(autouse=True)
def task_server(queues):
    # Make the function
    energy_func = partial(get_energy, method='pm7', basis=None)
    update_wrapper(energy_func, get_energy)

    # Make the task server
    config = Config(executors=[HighThroughputExecutor(max_workers=1)])
    server = ParslTaskServer([energy_func], queues, config)

    # Run and then kill when tests are complete
    server.start()
    yield server
    queues.send_kill_signal()
    server.join()


def test_exact(xyz_path, queues, tmpdir):
    # Make the thinker
    atoms = read(xyz_path)
    run_path = Path(tmpdir) / 'run'
    thinker = ExactHessianThinker(
        queues=queues,
        num_workers=1,
        atoms=atoms,
        run_dir=run_path,
    )
    assert run_path.exists()
    assert np.isnan(thinker.double_perturb).all()
    assert thinker.double_perturb.size == (3 ** 4 * 4)

    # Run it
    thinker.run()
    assert np.isfinite(thinker.single_perturb).all()

    # Make sure it picks up from a previous run
    thinker = ExactHessianThinker(
        queues=queues,
        num_workers=1,
        atoms=atoms,
        run_dir=run_path,
    )
    assert np.isfinite(thinker.single_perturb).all()
    assert np.isfinite(thinker.double_perturb).any()
    assert thinker.unperturbed_energy is not None

    # Make sure it doesn't do any new calculations
    start_size = (run_path / 'simulation-results.json').stat().st_size
    thinker.run()
    end_size = (run_path / 'simulation-results.json').stat().st_size
    assert start_size == end_size
