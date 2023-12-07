"""Use the debug queue on Theta"""
from parsl import Config, HighThroughputExecutor
from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
from parsl.addresses import address_by_interface


def make_config():
    return Config(
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='theta_local_htex_multinode',
                address=address_by_interface('vlan2360'),
                max_workers=1,
                provider=CobaltProvider(
                    queue='debug-flat-quad',
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 64 --cc depth"),
                    walltime='01:00:00',
                    nodes_per_block=2,
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    scheduler_options='#COBALT --attrs enable_ssh=1:filesystems=home,eagle',
                    # Command to be run before starting a worker, such as:
                    # 'module load Anaconda; source activate parsl_env'.
                    worker_init='''
module load miniconda-3
source activate /lus/eagle/projects/ExaLearn/faster-molecular-hessians/env
which python
realpath .
export PSIDATADIR=/lus/eagle/projects/ExaLearn/faster-molecular-hessians/env/share/psi4''',
                    cmd_timeout=120,
                ),
            )
        ],), 8, {'num_threads': 64, 'memory': '64GB'}
