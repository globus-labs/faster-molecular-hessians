"""Run on Logan's desktop computer"""
from parsl import Config, HighThroughputExecutor
from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
from parsl.addresses import address_by_interface


def make_config():
    return Config(
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='bettik',
                max_workers=1,
            )
        ]), 1, {'num_threads': 12, 'memory': '8GB',
                'command': "mpirun -n 2 /home/lward/Software/exachem/bin/bin/ExaChem exachem.json 1> exachem.out 2> exachem.err"}
