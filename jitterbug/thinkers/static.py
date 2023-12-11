"""Approach which uses a static set of structures to compute Hessian"""
from pathlib import Path

import ase
import numpy as np
from ase.db import connect
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import ResourceCounter, agent, result_processor

from .base import HessianThinker
from jitterbug.sampler.base import StructureSampler
from ..model.base import EnergyModel
from ..utils import read_from_string


class ApproximateHessianThinker(HessianThinker):
    """Approach which approximates a Hessian by computing it from a forcefield fit to structures

    Saves structures to an ASE db and labels them with the name of the sampler and the index
    """

    def __init__(self,
                 queues: ColmenaQueues,
                 num_workers: int,
                 atoms: ase.Atoms,
                 run_dir: Path,
                 sampler: StructureSampler,
                 num_to_run: int,
                 model: EnergyModel,
                 step_size: float = 0.005):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, atoms)
        self.step_size = step_size
        self.sampler = sampler
        self.num_to_run = num_to_run
        self.model = model

        # Generate the structures to be sampled
        self.to_sample = self.sampler.produce_structures(atoms, num_to_run)
        sampler_name = self.sampler.name
        self.logger.info(f'Generated {self.to_sample} structures with strategy: {sampler_name}')

        # Find how many we've done already
        self.db_path = self.run_dir / 'atoms.db'
        self.completed: set[int] = set()
        with connect(self.db_path) as db:
            for row in db.select(f'index<{self.num_to_run}', sampler=sampler_name):
                atoms = row.toatoms(True)
                ind = atoms.info['key_value_pairs']['index']
                assert np.isclose(atoms.positions, self.to_sample[ind].positions).all(), f'Structure {ind} in the DB and generated structure are inconsistent'
                self.completed.add(ind)
        num_remaining = self.num_to_run - len(self.completed)
        self.logger.info(f'Completed {len(self.completed)} structures already. Need to run {num_remaining} more')

    @agent()
    def submit_tasks(self):
        """Submit all required tasks then start the shutdown process by exiting"""

        for ind, atoms in enumerate(self.to_sample):
            # Skip structures which we've done already
            if ind in self.completed:
                continue

            # Submit it
            self.rec.acquire(None, 1)
            self.queues.send_inputs(
                atoms,
                method='get_energy',
                task_info={'index': ind}
            )

    @result_processor
    def store_energy(self, result: Result):
        """Store the energy in the appropriate files"""
        self.rec.release()

        # Store the result object to disk
        with self.result_file.open('a') as fp:
            print(result.json(exclude={'inputs'}), file=fp)

        if not result.success:
            self.logger.warning(f'Calculation failed due to {result.failure_info.exception}')
            return

        # Store the result into the ASE database
        sampler_name = self.sampler.name
        index = result.task_info['index']
        atoms = read_from_string(result.value, 'extxyz')
        assert np.isclose(result.args[0].positions, atoms.positions).all()
        self.completed.add(index)
        with connect(self.db_path) as db:
            db.write(atoms, sampler=sampler_name, index=index)
        self.logger.info(f'Saved completed structure. Progress: {len(self.completed)}/{self.num_to_run}'
                         f' ({len(self.completed) / self.num_to_run * 100:.2f}%)')

    def compute_hessian(self) -> np.ndarray:
        # Load the models
        atoms = []
        with connect(self.db_path) as db:
            for row in db.select(f'index<{self.num_to_run}', sampler=self.sampler.name):
                atoms.append(row.toatoms())
        self.logger.info(f'Pulled {len(atoms)} atoms for a training set')

        # Fit the model
        model = self.model.train(atoms)
        self.logger.info('Completed model fitting')

        return self.model.mean_hessian(model)
