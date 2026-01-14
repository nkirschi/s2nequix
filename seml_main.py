import jax

from jax.experimental.compilation_cache import compilation_cache
from nequix.train import train
from seml.experiment import Experiment

ex = Experiment()


@ex.config
def default_config():
    profile_device_memory = False  # noqa: F841
    jax_enable_x64 = False  # noqa: F841
    seed = 0  # noqa: F841


@ex.automain
def main(profile_device_memory: bool, jax_enable_x64: bool, overwrite: int):
    compilation_cache.set_cache_dir('./jax_cache')
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_enable_x64', bool(jax_enable_x64))
    jax.config.update('jax_default_matmul_precision', 'float32')
    jax.config.update('jax_explain_cache_misses', False)
    jax.config.update('jax_debug_nans', False)
    # grain.config.update('py_debug_mode', False)
    assert jax.default_backend() == 'gpu'

    train(config_path='configs/nequix.yml')
    if profile_device_memory:
        jax.profiler.save_device_memory_profile('memory_profile.prof')
