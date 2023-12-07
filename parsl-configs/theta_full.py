"""Use the backfill queue on Theta"""
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
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 64 --cc depth"),
                    walltime='0:30:00',
                    nodes_per_block=128,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=8,
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
        ],), 128 * 8 + 100, {'num_threads': 64, 'memory': '64GB'}
