import jax

from jax.experimental.compilation_cache import compilation_cache
import yaml
from nequix.train import _train
from seml.experiment import Experiment

ex = Experiment()


@ex.config
def default_config():
    profile_device_memory = False  # noqa: F841
    jax_enable_x64 = False  # noqa: F841
    config_path = None  # noqa: F841
    if config_path is not None:
        with open(config_path, "r") as f:
            config_override = yaml.safe_load(f)  # noqa: F841
        del f


@ex.automain
def main(
    profile_device_memory: bool, jax_enable_x64: bool, config_path: str, config_override: dict
):
    compilation_cache.set_cache_dir("./jax_cache")
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", bool(jax_enable_x64))
    jax.config.update("jax_default_matmul_precision", "float32")
    jax.config.update("jax_explain_cache_misses", False)
    jax.config.update("jax_debug_nans", False)
    # grain.config.update('py_debug_mode', False)
    assert jax.default_backend() == "gpu"
    config = {} | config_override  # need to copy because config_override is read-only
    _train(config=config)
    if profile_device_memory:
        jax.profiler.save_device_memory_profile("memory_profile.prof")
