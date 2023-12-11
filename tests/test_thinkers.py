from functools import partial, update_wrapper
from pathlib import Path

import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from colmena.queue.python import PipeQueues
from colmena.task_server.parsl import ParslTaskServer
from parsl import Config, HighThroughputExecutor
from pytest import fixture, mark

from jitterbug.compare import compare_hessians
from jitterbug.model.dscribe import make_global_mbtr_model
from jitterbug.parsl import get_energy
from jitterbug.sampler import UniformSampler
from jitterbug.thinkers.exact import ExactHessianThinker
from jitterbug.thinkers.static import ApproximateHessianThinker
from jitterbug.utils import make_calculator


@fixture()
def queues():
    return PipeQueues(topics='simulation')


@fixture(params=['water.xyz', 'radical.extxyz'])
def atoms(file_dir, request):
    return read(file_dir / request.param)


@fixture()
def ase_hessian(atoms, tmp_path) -> np.ndarray:
    atoms.calc = make_calculator('pm7', None)
    vib = Vibrations(atoms, delta=0.005, name=str(Path(tmp_path) / 'vib'))
    vib.run()
    return vib.get_vibrations().get_hessian_2d()


@fixture()
def mbtr(atoms):
    return make_global_mbtr_model(atoms)


@fixture(autouse=True)
def task_server(queues):
    # Make the function
    energy_func = partial(get_energy, method='pm7', basis=None)
    update_wrapper(energy_func, get_energy)

    # Make the task server
    config = Config(executors=[HighThroughputExecutor(max_workers=2)])
    server = ParslTaskServer([energy_func], queues, config)

    # Run and then kill when tests are complete
    server.start()
    yield server
    queues.send_kill_signal()
    server.join()


@mark.timeout(60)
def test_exact(atoms, queues, tmpdir, ase_hessian):
    # Make the thinker
    run_path = Path(tmpdir) / 'run'
    thinker = ExactHessianThinker(
        queues=queues,
        num_workers=2,
        atoms=atoms,
        run_dir=run_path,
    )
    assert run_path.exists()
    assert np.isnan(thinker.double_perturb).all()
    assert thinker.double_perturb.size == len(atoms) ** 2 * 9 * 4

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

    # Compute the Hessian
    hessian = thinker.compute_hessian()
    assert np.isfinite(hessian).all()
    assert hessian.shape == (len(atoms) * 3, len(atoms) * 3)

    # Make sure it is close to ase's
    comparison = compare_hessians(atoms, hessian, ase_hessian)
    assert abs(comparison.zpe_error) < 0.2


@mark.timeout(60)
def test_approx(atoms, queues, tmpdir, ase_hessian, mbtr):
    # Make the thinker
    run_path = Path(tmpdir) / 'run'
    thinker = ApproximateHessianThinker(
        queues=queues,
        num_workers=2,
        atoms=atoms,
        run_dir=run_path,
        num_to_run=128,
        model=mbtr,
        sampler=UniformSampler()
    )
    assert run_path.exists()
    assert thinker.completed == set()

    # Run it
    thinker.run()
    assert len(thinker.completed) == 128

    # Make sure it picks up from a previous run
    thinker = ApproximateHessianThinker(
        queues=queues,
        num_workers=2,
        atoms=atoms,
        run_dir=run_path,
        num_to_run=128,
        model=mbtr,
        sampler=UniformSampler()
    )
    assert len(thinker.completed) == 128

    # Make sure it doesn't do any new calculations
    start_size = (run_path / 'simulation-results.json').stat().st_size
    thinker.run()
    end_size = (run_path / 'simulation-results.json').stat().st_size
    assert start_size == end_size

    # Compute the Hessian
    hessian = thinker.compute_hessian()
    assert np.isfinite(hessian).all()
    assert hessian.shape == (len(atoms) * 3, len(atoms) * 3)
