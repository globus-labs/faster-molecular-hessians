import ase


class StructureSampler:
    """Base class for generating structures used to train Hessian model

    Options for the sampler should be defined in the initializer.
    """

    @property
    def name(self) -> str:
        """Name for the sampling strategy"""
        raise NotImplementedError()

    def produce_structures(self, atoms: ase.Atoms, count: int, seed: int = 1) -> list[ase.Atoms]:
        """Generate a set of training structure

        Args:
            atoms: Unperturbed geometry
            count: Number of structure to produce
            seed: Random seed
        Returns:
            List of structures to be evaluated
        """

        raise NotImplementedError()
