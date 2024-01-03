"""Use the debug queue on Theta"""
from parsl import Config, HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import SimpleLauncher
from parsl.addresses import address_by_interface

user_opts = {
    "worker_init": '''

# Exachem Env
module use /soft/modulefiles/
module load spack-pe-gcc/0.4-rc1 numactl/2.0.14-gcc-testing cmake
module load oneapi/release/2023.12.15.001
#module load intel_compute_runtime/release/stable-736.25
module list

export MPIR_CVAR_ENABLE_GPU=0
export FI_CXI_DEFAULT_CQ_SIZE=131072
export MEMKIND_HBW_THRESHOLD=402400
export FI_CXI_CQ_FILL_PERCENT=20
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x001
export ZES_ENABLE_SYSMAN=1
export TAMM_USE_MEMKIND=1

export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1

unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE

# ExaChem Paths
export TAMMDIR=/lus/gecko/projects/CSC249ADCD08_CNDA/software/exachem/install/tamm
export BINDSH=/lus/gecko/projects/CSC249ADCD08_CNDA/software/bind-tiles-closest.sh
export LD_LIBRARY_PATH=$TAMMDIR/lib64:$LD_LIBRARY_PATH
export EXE=$TAMMDIR/bin/ExaChem

# Load conda
source activate /lus/gecko/projects/CSC249ADCD08_CNDA/faster-molecular-hessians/env-cpu
hostname
which python
realpath .''',
    "scheduler_options":"" ,
    "account":          "CSC249ADCD08_CNDA",
    "queue":            "EarlyAppAccess",
    "walltime":         "12:00:00",
    "nodes_per_block":  4, # think of a block as one job on sunspot
    "cpus_per_node":    208, # this is the number of threads available on one sunspot node
}

# Configure the launch command
num_nodes = user_opts['nodes_per_block']
ranks_per_node = 13
total_ranks = num_nodes * ranks_per_node
exachem_cmd = f'mpiexec -n {total_ranks} --ppn {ranks_per_node} --spindle --pmi=pmix --cpu-bind list:2:10:18:26:34:42:58:66:74:82:90:98:99 --mem-bind list:0:0:0:0:0:0:1:1:1:1:1:1:1 --env TAMM_USE_MEMKIND=1 --env OMP_NUM_THREADS=1 $BINDSH $EXE exachem.json 1>exachem.out 2>exachem.err'


def make_config():
    return Config(
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="sunspot_test",
                #                address=address_by_interface(),
                max_workers=1,
                provider=PBSProProvider(
                    account=user_opts["account"],
                    queue=user_opts["queue"],
                    worker_init=user_opts["worker_init"],
                    walltime=user_opts["walltime"],
                    scheduler_options=user_opts["scheduler_options"],
                    launcher=SimpleLauncher(),
                    nodes_per_block=user_opts["nodes_per_block"],
                    min_blocks=0,
                    max_blocks=1, # Can increase more to have more parallel batch jobs
                    cpus_per_node=user_opts["cpus_per_node"],
                )
           )
            ],), 8, {'command': exachem_cmd, 'scratch_dir': '/lus/gecko/projects/CSC249ADCD08_CNDA/faster-molecular-hessians/exachem-output', 'cleanup': False}
