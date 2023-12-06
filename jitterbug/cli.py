"""Command line interface to running Jitterbug on a new molecule"""
from argparse import ArgumentParser
from functools import partial, update_wrapper
from pathlib import Path
from typing import Optional
import logging
import sys

import numpy as np
from ase.io import read
from colmena.queue import PipeQueues
from colmena.task_server import ParslTaskServer
from parsl import Config, HighThroughputExecutor

from jitterbug.parsl import get_energy
from jitterbug.thinkers.exact import ExactHessianThinker

logger = logging.getLogger(__name__)


def main(args: Optional[list[str]] = None):
    """Run Jitterbug"""

    parser = ArgumentParser()
    parser.add_argument('xyz', help='Path to the XYZ file')
    parser.add_argument('--method', nargs=2, required=True,
                        help='Method to use to compute energies. Format: [method] [basis]. Example: B3LYP 6-31g*')
    parser.add_argument('--exact', help='Compute Hessian using numerical derivatives', action='store_true')
    args = parser.parse_args(args)

    # Load the structure
    xyz_path = Path(args.xyz)
    atoms = read(args.xyz)
    xyz_name = xyz_path.with_suffix('').name

    # Make the run directory
    method, basis = (x.lower() for x in args.method)
    run_dir = Path('run') / xyz_name / f'{method}_{basis}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Start logging
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / 'run.log', mode='a')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    for logger_name in ['jitterbug']:
        my_logger = logging.getLogger(logger_name)
        for handler in handlers:
            my_logger.addHandler(handler)
        my_logger.setLevel(logging.INFO)

    # Write the XYZ file to the run directory
    if (run_dir / xyz_path.name).exists() and (run_dir / xyz_path.name).read_text() != xyz_path.read_text():
        raise ValueError('Run exists for a different structure with the same name.')
    (run_dir / xyz_path.name).write_text(xyz_path.read_text())
    logger.info(f'Started run for {xyz_name} at {method}/{basis}. Run directory: {run_dir.absolute()}')

    # Make the function to compute energy
    energy_fun = partial(get_energy, method=method, basis=basis)
    update_wrapper(energy_fun, get_energy)

    # Create a thinker
    queues = PipeQueues(topics=['simulation'])
    if args.exact:
        thinker = ExactHessianThinker(
            queues=queues,
            atoms=atoms,
            run_dir=run_dir,
            num_workers=1,
        )
        functions = []  # No other functions to run
    else:
        raise NotImplementedError()

    # Create the task server
    config = Config(run_dir=str(run_dir / 'parsl-logs'), executors=[HighThroughputExecutor(max_workers=1)])
    task_server = ParslTaskServer([energy_fun] + functions, queues, config)

    # Run everything
    try:
        task_server.start()
        thinker.run()
    finally:
        queues.send_kill_signal()

    # Get the Hessian
    hessian = thinker.compute_hessian()
    hess_path = run_dir / 'hessian.npy'
    np.save(hess_path, hessian)
    logger.info(f'Wrote Hessian to {hess_path}')
