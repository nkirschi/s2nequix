import argparse
import functools
import os
import time
from collections import defaultdict
from pathlib import Path

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as onp
import optax
import yaml
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from nequix.data import (
    DataLoader,
    AseDBDataset,
    IndexDataset,
    ParallelLoader,
    SubepochalLoader,
    average_atom_energies,
    dataset_stats,
    prefetch,
)
from nequix.early_stopping import EarlyStopping
from nequix.model import Nequix, save_model, weight_decay_mask


@eqx.filter_jit
def loss(model, batch, energy_weight, force_weight, stress_weight, loss_type="huber"):
    """Return huber loss and MAE of energy and force in eV and eV/Å respectively"""
    energy, forces, stress = model(batch)
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    config = {
        "mse": {"energy": "mse", "force": "mse", "stress": "mse"},
        "huber": {"energy": "huber", "force": "huber", "stress": "huber"},
        "mae": {"energy": "mae", "force": "l2", "stress": "mae"},
    }[loss_type]

    loss_fns = {
        "mae": lambda pred, true: jnp.abs(pred - true),
        "mse": lambda pred, true: (pred - true) ** 2,
        "huber": lambda pred, true: optax.losses.huber_loss(pred, true, delta=0.1),
    }

    # energy per atom (see eq. 30 https://www.nature.com/articles/s41467-023-36329-y)
    # can be achieved by dividing predictied and true energy by number of atoms
    energy_loss_per_atom = jnp.sum(
        loss_fns[config["energy"]](energy / batch.n_node, batch.globals["energy"] / batch.n_node)
        * graph_mask
    ) / jnp.sum(graph_mask)

    if config["force"] == "l2":
        # l2 norm loss for forces
        # NOTE: double where trick is needed to avoid nan's
        force_diff_squared = jnp.sum((forces - batch.nodes["forces"]) ** 2, axis=-1)
        safe_force_diff_squared = jnp.where(force_diff_squared == 0.0, 1.0, force_diff_squared)
        force_loss = jnp.sum(
            jnp.where(force_diff_squared == 0.0, 0.0, jnp.sqrt(safe_force_diff_squared)) * node_mask
        ) / jnp.sum(node_mask)
    else:
        force_loss = jnp.sum(
            loss_fns[config["force"]](forces, batch.nodes["forces"]) * node_mask[:, None]
        ) / (3 * jnp.sum(node_mask))

    if stress_weight > 0:
        stress_loss = jnp.sum(
            loss_fns[config["stress"]](stress, batch.globals["stress"]) * graph_mask[:, None, None]
        ) / (9 * jnp.sum(graph_mask))
    else:
        stress_loss = 0

    total_loss = (
        energy_weight * energy_loss_per_atom
        + force_weight * force_loss
        + stress_weight * stress_loss
    )

    # metrics:

    # MAE energy
    energy_mae_per_atom = jnp.sum(
        jnp.abs(energy / batch.n_node - batch.globals["energy"] / batch.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    # MAE forces
    force_mae = jnp.sum(jnp.abs(forces - batch.nodes["forces"]) * node_mask[:, None]) / (
        3 * jnp.sum(node_mask)
    )

    # MAE stress
    if stress_weight > 0:
        stress_mae_per_atom = jnp.sum(
            jnp.abs(stress - batch.globals["stress"])
            / jnp.where(batch.n_node > 0, batch.n_node, 1.0)[:, None, None]
            * graph_mask[:, None, None]
        ) / (9 * jnp.sum(graph_mask))
    else:
        stress_mae_per_atom = 0.0 * energy_mae_per_atom

    return total_loss, {
        "energy_mae_per_atom": energy_mae_per_atom,
        "force_mae": force_mae,
        "stress_mae_per_atom": stress_mae_per_atom,
    }


def evaluate(
    model, dataloader, energy_weight=1.0, force_weight=1.0, stress_weight=1.0, loss_type="huber"
):
    """Return loss and RMSE of energy and force in eV and eV/Å respectively"""
    total_metrics = defaultdict(int)
    total_count = 0
    for batch in prefetch(dataloader):
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        val_loss, metrics = loss(
            model, batch, energy_weight, force_weight, stress_weight, loss_type
        )
        total_metrics["loss"] += val_loss * n_graphs
        for key, value in metrics.items():
            total_metrics[key] += value * n_graphs
        total_count += n_graphs

    for key, value in total_metrics.items():
        total_metrics[key] = value / total_count

    return total_metrics


def save_training_state(
    path, model, ema_model, optim, opt_state, step, epoch, best_val_loss, wandb_run_id=None
):
    state = {
        "model": model,
        "ema_model": ema_model,
        "optim": optim,
        "opt_state": opt_state,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb_run_id,
    }
    with open(path, "wb") as f:
        cloudpickle.dump(state, f)


def load_training_state(path):
    with open(path, "rb") as f:
        state = cloudpickle.load(f)
    return (
        state["model"],
        state["ema_model"],
        state["optim"],
        state["opt_state"],
        state["step"],
        state["epoch"],
        state["best_val_loss"],
        state.get("wandb_run_id"),
    )


def print_summary(model):
    print("--- Model Structure ---")
    eqx.tree_pprint(model)

    # Filter for arrays (weights/biases) and ignore static configuration
    params = [x for x in jax.tree_util.tree_leaves(model) if eqx.is_array(x)]
    count = sum(x.size for x in params)

    # Calculate size in MB (assuming float32 = 4 bytes)
    size_mb = count * 4 / (1024 * 1024)

    print("\n--- Model Stats ---")
    print(f"Total Parameters: {count:,}")
    print(f"Model Size:       {size_mb:.2f} MB")


def train(config_path: str):
    """Train a Nequix model from a config file. See configs/nequix-mp-1.yaml for an example."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    _train(config)


def _train(config: dict, run_notes: str = ""):
    # use TMPDIR for slurm jobs if available
    if "schedule" not in config:
        config["schedule"] = "cosine"
    if "early_stopping_patience" not in config:
        config["early_stopping_patience"] = int(1e20)
    if "early_stopping_min_relative_improvement" not in config:
        config["early_stopping_min_relative_improvement"] = 0.0
    if "molsize_range" not in config:
        config["molsize_range"] = None
    if "dataset_name" not in config:
        config["dataset_name"] = None
    if "subset" not in config:
        config["subset"] = None
    if "valid_path" not in config:
        config["valid_path"] = None

    wandb_init_kwargs = {"project": "nequix", "config": config, "notes": run_notes}
    using_checkpoint = "resume_from" in config and Path(config["resume_from"]).exists()
    if using_checkpoint:
        (
            model,
            ema_model,
            optim,
            opt_state,
            step,
            start_epoch,
            best_val_loss,
            wandb_run_id,
        ) = load_training_state(config["resume_from"])
        wandb_init_kwargs.update({"id": wandb_run_id, "resume": "allow"})

    wandb.init(**wandb_init_kwargs)
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb_run_id = getattr(wandb.run, "id", None)
        try:
            run_suffix = f"-{wandb.run.name.split('-')[-1]}"
        except IndexError:
            run_suffix = ""
        wandb.run.name = f"{wandb_run_id}{run_suffix}"

    checkpoint_path = Path(config["checkpoint_dir"]) / str(wandb_run_id)
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f"loading training dataset from {config['train_path']}...")
    train_dataset = AseDBDataset(
        file_path=config["train_path"],
        atomic_numbers=config["atomic_numbers"],
        cutoff=config["cutoff"],
        backend="jax",
        load_spectral=(config["spectral_layer_type"] is not None)
        if "spectral_layer_type" in config
        else False,
        laplacian_cutoff_interval=tuple(config["laplacian_cutoff_interval"])
        if "laplacian_cutoff_interval" in config
        else None,
        num_eigenvectors=config["num_eigenvectors"] if "num_eigenvectors" in config else None,
    )

    if config["valid_path"] is not None:
        print(f"loading validation dataset from {config['valid_path']}...")
        val_dataset = AseDBDataset(
            file_path=config["valid_path"],
            atomic_numbers=config["atomic_numbers"],
            cutoff=config["cutoff"],
            backend="jax",
        )
    else:
        assert "valid_frac" in config, "valid_frac must be specified if valid_path is not provided"
        print(f"splitting training dataset with valid_frac={config['valid_frac']} ...")
        train_dataset, val_dataset = train_dataset.split(valid_frac=config["valid_frac"])

    # optional filtering
    train_mask = onp.ones(len(train_dataset), dtype=bool)
    val_mask = onp.ones(len(val_dataset), dtype=bool)
    if config["valid_path"] is not None:
        train_meta = onp.load(f"{config['train_path']}/metadata.npz")
        val_meta = onp.load(f"{config['valid_path']}/metadata.npz")
    else:
        meta = onp.load(f"{config['train_path']}/metadata.npz")
        train_meta = {k: v[train_dataset.indices] for k, v in meta.items()}
        val_meta = {k: v[val_dataset.indices] for k, v in meta.items()}
    stats_string = "stats"

    if config["subset"] is not None:
        print(f"filtering dataset to subset={config['subset']} ...")
        train_mask &= train_meta["data_ids"] == config["subset"]
        val_mask &= val_meta["data_ids"] == config["subset"]
        stats_string += f"_{config['subset']}"

    if config["molsize_range"] is not None:
        print(f"filtering dataset to molsize_range={config['molsize_range']} ...")
        min_n, max_n = config["molsize_range"]
        train_mask &= (train_meta["natoms"] >= min_n) & (train_meta["natoms"] <= max_n)
        val_mask &= (val_meta["natoms"] >= min_n) & (val_meta["natoms"] <= max_n)
        stats_string += f"_{min_n}-{max_n}atoms"

    train_idx = onp.argwhere(train_mask).squeeze()
    val_idx = onp.argwhere(val_mask).squeeze()
    train_dataset = IndexDataset(train_dataset, train_idx)
    val_dataset = IndexDataset(val_dataset, val_idx)

    stats_path = f"{config['train_path']}/{stats_string}.npz"
    if os.path.exists(stats_path):
        print(f"loading dataset statistics from {stats_path} ...")
        stats = onp.load(stats_path)
    else:
        print("computing dataset statistics ...")
        atom_energies = average_atom_energies(train_dataset)
        stats = dataset_stats(train_dataset, atom_energies)
        stats["atom_energies"] = atom_energies
        onp.savez(stats_path, **stats)

    print(f"dataset sizes (train/val): {len(train_dataset)}/{len(val_dataset)}")
    for key, val in stats.items():
        print(f"{key}: {val}")

    num_devices = len(jax.devices())
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=16,
    )
    if "subepoch_length" in config and config["subepoch_length"] is not None:
        train_loader = SubepochalLoader(train_loader, length=config["subepoch_length"])
        print(f"using subepochs of {config['subepoch_length']} batches")
    else:
        print("using full epochs")
    train_loader = ParallelLoader(train_loader, num_devices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=16,
    )

    if not using_checkpoint:
        key = jax.random.key(config["seed"] if "seed" in config else 0)
        model = Nequix(
            key,
            n_species=len(config["atomic_numbers"]),
            hidden_irreps=config["hidden_irreps"],
            lmax=config["lmax"],
            cutoff=config["cutoff"],
            n_layers=config["n_layers"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
            index_weights=config["index_weights"],
            layer_norm=config["layer_norm"],
            shift=stats["shift"],
            scale=stats["scale"],
            avg_n_neighbors=stats["avg_n_neighbors"],
            atom_energies=stats["atom_energies"],
            spectral_layer_type=config["spectral_layer_type"]
            if "spectral_layer_type" in config
            else None,
        )
    print_summary(model)

    # NB: this is not exact because of dynamic batching but should be close enough
    steps_per_epoch = len(train_dataset) // (config["batch_size"] * jax.device_count())
    match config["schedule"]:
        case "cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=config["learning_rate"] * config["warmup_factor"],
                peak_value=config["learning_rate"],
                end_value=1e-6,
                warmup_steps=config["warmup_epochs"] * steps_per_epoch,
                decay_steps=config["n_epochs"] * steps_per_epoch,
            )
            # effectively disable plateau reducer since we're using cosine schedule
            plateau_reducer = optax.contrib.reduce_on_plateau(patience=0, factor=1.0)
        case "plateau":
            assert "plateau_patience" in config, (
                "plateau_patience must be specified for plateau schedule"
            )
            assert "plateau_factor" in config, (
                "plateau_factor must be specified for plateau schedule"
            )
            schedule = optax.warmup_constant_schedule(
                init_value=config["learning_rate"] * config["warmup_factor"],
                peak_value=config["learning_rate"],
                warmup_steps=config["warmup_epochs"] * steps_per_epoch,
            )
            plateau_reducer = optax.contrib.reduce_on_plateau(
                patience=config["plateau_patience"],
                factor=config["plateau_factor"],
            )
        case _:
            raise ValueError(f"learning rate schedule {config['schedule']} not supported")

    plateau_state = plateau_reducer.init(None)
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_relative_improvement=config["early_stopping_min_relative_improvement"],
    )

    if not using_checkpoint:
        match config["optimizer"]:
            case "adamw":
                optim = optax.adamw(
                    learning_rate=schedule,
                    weight_decay=config["weight_decay"],
                    mask=weight_decay_mask(model),
                )
            case "muon":
                optim = optax.contrib.muon(
                    learning_rate=schedule,
                    weight_decay=config["weight_decay"] if config["weight_decay"] != 0.0 else None,
                    weight_decay_mask=weight_decay_mask(model),
                )
            case _:
                raise ValueError(f"optimizer {config['optimizer']} not supported")

        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optim,
        )
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        model = jax.device_put_replicated(model, list(jax.devices()))
        opt_state = jax.device_put_replicated(opt_state, list(jax.devices()))
        ema_model = jax.tree.map(lambda x: x.copy(), model)  # copy model
        step = jnp.array(0)
        start_epoch = 0
        best_val_loss = float("inf")

    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])
    wandb.run.summary["param_count"] = param_count
    wandb_sync = (
        TriggerWandbSyncHook() if os.environ.get("WANDB_MODE") == "offline" else lambda: None
    )

    # @eqx.filter_jit
    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, None, 0, 0, None), axis_name="device")
    def train_step(model, ema_model, step, opt_state, batch, lr_scale):
        # training step
        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model,
            batch,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        updates = jax.tree.map(lambda x: lr_scale * x, updates)  # plateau handling factor
        model = eqx.apply_updates(model, updates)

        # update EMA model
        # don't weight early steps as much (from https://github.com/fadel/pytorch_ema)
        decay = jnp.minimum(config["ema_decay"], (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)

        return (
            model,
            ema_model,
            opt_state,
            total_loss,
            metrics,
        )

    for epoch in range(start_epoch, config["n_epochs"]):
        start_time = time.time()
        train_loader.loader.set_epoch(epoch)
        for batch in prefetch(train_loader):
            batch_time = time.time() - start_time
            start_time = time.time()
            (model, ema_model, opt_state, total_loss, metrics) = train_step(
                model, ema_model, step, opt_state, batch, plateau_state.scale
            )
            train_time = time.time() - start_time
            step = step + 1
            if step % config["log_every"] == 0:
                logs = {}
                logs["train/loss"] = total_loss.mean().item()
                logs["learning_rate"] = schedule(step).item() * plateau_state.scale
                logs["train/batch_time"] = batch_time
                logs["train/train_time"] = train_time
                for key, value in metrics.items():
                    logs[f"train/{key}"] = value.mean().item()
                logs["train/batch_size"] = (
                    jax.vmap(jraph.get_graph_padding_mask)(batch).sum().item()
                )
                wandb.log(logs, step=step)
                print(f"step: {step}, logs: {logs}")
                wandb_sync()
            start_time = time.time()

        ema_model_single = jax.tree.map(lambda x: x[0], ema_model)
        val_metrics = evaluate(
            ema_model_single,
            val_loader,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            for path in [Path(wandb.run.dir), Path(checkpoint_path)]:
                save_model(path / "model.nqx", ema_model_single, config)
                save_training_state(
                    path / "state.pkl",
                    model,
                    ema_model,
                    optim,
                    opt_state,
                    step,
                    epoch + 1,
                    best_val_loss,
                    wandb_run_id=wandb_run_id,
                )

        logs = {}
        for key, value in val_metrics.items():
            logs[f"val/{key}"] = value.item()
        logs["epoch"] = epoch
        wandb.log(logs, step=step)
        print(f"epoch: {epoch}, logs: {logs}")
        wandb_sync()

        _, plateau_state = plateau_reducer.update(
            updates=None,
            state=plateau_state,
            value=val_metrics["loss"],
        )

        if early_stopping.stop(val_metrics["loss"]):
            print(f">>> EARLY STOPPING at epoch {epoch} <<<")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)


if __name__ == "__main__":
    main()
