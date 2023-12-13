"""Run an exact Hessian computation"""
from csv import reader, writer
from pathlib import Path
from typing import Optional

import ase
import numpy as np
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import ResourceCounter, agent, result_processor

from jitterbug.thinkers.base import HessianThinker
from jitterbug.utils import read_from_string


class ExactHessianThinker(HessianThinker):
    """Schedule the calculation of a complete set of numerical derivatives"""

    def __init__(self, queues: ColmenaQueues, num_workers: int, atoms: ase.Atoms, run_dir: Path, step_size: float = 0.005):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, atoms)

        # Initialize storage for the energies
        self.step_size = step_size
        self.unperturbed_energy: Optional[float] = None
        self.single_perturb = np.zeros((len(atoms), 3, 2)) * np.nan  # Perturbation of a single direction. [atom_id, axis (xyz), dir_id (0 back, 1 forward)]
        self.double_perturb = np.zeros((len(atoms), 3, 2, len(atoms), 3, 2)) * np.nan
        self.total_required = len(atoms) * 3 * 2 + (len(atoms) * 3) * (len(atoms) * 3 - 1) * 2 + 1
        # Perturbation of two directions [atom1_id, axis1, dir1_id, atom2_id, axis2 dir2_id]
        self.logger.info(f'Preparing to run {self.total_required} energy computations')

        # Load what has been run already
        self.total_complete = 0
        self.energy_path = self.run_dir / 'unperturbed.energy'
        self.single_path = self.run_dir / 'single_energies.csv'
        self.double_path = self.run_dir / 'double_energies.csv'

        if self.energy_path.exists():
            self.unperturbed_energy = float(self.energy_path.read_text())
            self.total_complete += 1

        if self.single_path.exists():
            with self.single_path.open() as fp:
                count = 0
                for row in reader(fp):
                    index = tuple(map(int, row[:-1]))
                    count += 1
                    self.single_perturb[index] = row[-1]
            self.total_complete += count
            self.logger.info(f'Read {count} single perturbations out of {self.single_perturb.size}')

        if self.double_path.exists():
            with self.double_path.open() as fp:
                count = 0
                for row in reader(fp):
                    count += 1
                    # Get the index and its symmetric counterpart
                    index = tuple(map(int, row[:-1]))
                    sym_index = list(index)
                    sym_index[:3], sym_index[3:] = index[3:], index[:3]
                    self.double_perturb[index] = self.double_perturb[tuple(sym_index)] = row[-1]
                to_do = (len(atoms) * 3) * (len(atoms) * 3 - 1) * 2
            self.total_complete += count
            self.logger.info(f'Read {count} double perturbations out of {to_do}')

    @agent()
    def submit_tasks(self):
        """Submit all required tasks then start the shutdown process by exiting"""

        # Start with the unperturbed energy
        if self.unperturbed_energy is None:
            self.rec.acquire(None, 1)
            self.queues.send_inputs(
                self.atoms,
                method='get_energy',
                task_info={'type': 'unperturbed'}
            )

        # Submit the single perturbations
        with np.nditer(self.single_perturb, flags=['multi_index']) as it:
            count = 0
            for x in it:
                # Skip if done
                if np.isfinite(x):
                    continue

                # Submit if not done
                self.rec.acquire(None, 1)  # Wait until resources are free
                count += 1
                atom_id, axis_id, dir_id = it.multi_index

                new_atoms = self.atoms.copy()
                new_atoms.positions[atom_id, axis_id] += self.step_size - 2 * self.step_size * dir_id
                self.queues.send_inputs(
                    new_atoms,
                    method='get_energy',
                    task_info={'type': 'single', 'coord': it.multi_index}
                )
        self.logger.info(f'Finished submitting {count} single perturbations')

        # Submit the double perturbations
        with np.nditer(self.double_perturb, flags=['multi_index']) as it:
            count = 0
            for x in it:
                # Skip if done
                if np.isfinite(x):
                    continue

                # Skip if perturbing the same direction twice, or if from the lower triangle
                if it.multi_index[:2] == it.multi_index[3:5] or it.multi_index[:3] < it.multi_index[3:]:
                    continue

                # Submit if not done
                self.rec.acquire(None, 1)
                count += 1

                # Perturb two axes
                new_atoms = self.atoms.copy()
                for atom_id, axis_id, dir_id in [it.multi_index[:3], it.multi_index[3:]]:
                    new_atoms.positions[atom_id, axis_id] += self.step_size - 2 * self.step_size * dir_id

                self.queues.send_inputs(
                    new_atoms,
                    method='get_energy',
                    task_info={'type': 'double', 'coord': it.multi_index}
                )
        self.logger.info(f'Finished submitting {count} double perturbations')

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

        calc_type = result.task_info['type']
        atoms = read_from_string(result.value, 'extxyz')
        energy = atoms.get_potential_energy()
        self.total_complete += 1

        # Store unperturbed energy
        if calc_type == 'unperturbed':
            self.logger.info('Storing energy of unperturbed structure')
            self.unperturbed_energy = energy
            self.energy_path.write_text(str(energy))
            return

        # Store perturbed energy
        coord = result.task_info['coord']
        self.logger.info(f'Saving a {calc_type} perturbation: ({",".join(map(str, coord))}).'
                         f' Progress: {self.total_complete}/{self.total_required}'
                         f' ({self.total_complete/self.total_required * 100:.2f}%)')
        if calc_type == 'single':
            energy_file = self.single_path
            energies = self.single_perturb
        else:
            energy_file = self.double_path
            energies = self.double_perturb

        with energy_file.open('a') as fp:
            csv_writer = writer(fp)
            csv_writer.writerow(coord + [energy])

        energies[tuple(coord)] = energy
        if calc_type == 'double':
            sym_coord = list(coord)
            sym_coord[:3], sym_coord[3:] = coord[3:], coord[:3]
            energies[tuple(sym_coord)] = energy

    def compute_hessian(self) -> np.ndarray:
        """Compute the Hessian using finite differences

        Returns:
            Hessian in the 2D form
        Raises:
            (ValueError) If there is missing data
        """

        # Check that all data are available
        n_atoms = len(self.atoms)
        if not np.isfinite(self.single_perturb).all():
            raise ValueError(f'Missing {np.isnan(self.single_perturb).sum()} single perturbations')
        expected_double = (n_atoms * 3) * (n_atoms * 3 - 1) * 4
        if not np.isfinite(self.double_perturb).sum() == expected_double:
            raise ValueError(f'Missing {expected_double - np.isfinite(self.double_perturb).sum()} double perturbations')

        # Flatten the arrays
        single_flat = np.reshape(self.single_perturb, (n_atoms * 3, 2))
        double_flat = np.reshape(self.double_perturb, (n_atoms * 3, 2, n_atoms * 3, 2))

        # Compute the finite differences
        #  https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences
        diagonal = (single_flat.sum(axis=1) - self.unperturbed_energy * 2) / (self.step_size ** 2)
        off_diagonal = (double_flat[:, 0, :, 0] + double_flat[:, 1, :, 1] - double_flat[:, 0, :, 1] - double_flat[:, 1, :, 0]) / (4 * self.step_size ** 2)
        np.fill_diagonal(off_diagonal, 0)
        return np.diag(diagonal) + off_diagonal
