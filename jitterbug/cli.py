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

from jitterbug.parsl import get_energy, load_configuration
from jitterbug.thinkers.exact import ExactHessianThinker
from jitterbug.utils import make_calculator
from jitterbug.sampler import methods

logger = logging.getLogger(__name__)

approx_methods = list(methods.keys())


def main(args: Optional[list[str]] = None):
    """Run Jitterbug"""

    parser = ArgumentParser()
    parser.add_argument('xyz',
                        help='Path to the XYZ file. Use extended XYZ format '
                             'to store information about charged or radical molecules')
    parser.add_argument('--method', nargs=2, required=True,
                        help='Method to use to compute energies. Format: [method] [basis]. Example: B3LYP 6-31g*')
    parser.add_argument('--approach', default='exact',
                        choices=['exact', 'static'],
                        help='Method used to compute the Hessian. Either "exact", or'
                             '"static" for a approximate method using a fixed set of structures.')
    parser.add_argument('--amount-to-run', default=0.1, type=float,
                        help='Amount of structures to run for approximate Hessian. '
                             'Either number (if >1) or fraction of number required to compute exact Hessian,'
                             '`6N + 2 * (3N) * (3N - 1) + 1`.')
    parser.add_argument('--parsl-config', help='Path to the Parsl configuration to use')
    args = parser.parse_args(args)

    # Load the structure
    xyz_path = Path(args.xyz)
    atoms = read(args.xyz, format='extxyz')
    xyz_name = xyz_path.with_suffix('').name

    # Make the run directory
    method, basis = (x.lower() for x in args.method)
    compute_name = args.approach
    run_dir = Path('run') / xyz_name / f'{method}_{basis}_{compute_name}'
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

    # Load Parsl configuration
    if args.parsl_config is None:
        config = Config(run_dir=str(run_dir / 'parsl-logs'), executors=[HighThroughputExecutor(max_workers=1)])
        num_workers = 1
        ase_options = {}
        logger.info('Running computations locally, one-at-a-time')
    else:
        config, num_workers, ase_options = load_configuration(args.parsl_config)
        logger.info(f'Running on {num_workers} workers as defined by {args.parsl_config}')

    # Add multiplicity to the options
    if atoms.get_initial_magnetic_moments().sum() > 0:
        mult = int(atoms.get_initial_magnetic_moments().sum()) + 1
        ase_options['multiplicity'] = int(mult)
        logger.info(f'Running with a multiplicity of {mult}')

        # Test making the calculator
        calc = make_calculator(method, basis, **ase_options)
        if calc.name == 'psi4':
            atoms.set_initial_magnetic_moments([0] * len(atoms))
            ase_options['charge'] = int(atoms.get_initial_charges().sum())
            logger.info('Using Psi4: Removed charge and magmom information from atoms object.')

    # Make the function to compute energy
    energy_fun = partial(get_energy, method=method, basis=basis, **ase_options)
    update_wrapper(energy_fun, get_energy)

    # Create a thinker
    queues = PipeQueues(topics=['simulation'])
    if args.exact:
        thinker = ExactHessianThinker(
            queues=queues,
            atoms=atoms,
            run_dir=run_dir,
            num_workers=num_workers,
        )
        functions = []  # No other functions to run
    else:
        raise NotImplementedError()

    # Create the task server
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
