from parsl import Config, HighThroughputExecutor


def make_config():
    return Config(executors=[HighThroughputExecutor(max_workers=1)]), 1, {}
