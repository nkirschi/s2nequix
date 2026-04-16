"""MACE model ported to equinox + e3nn_jax.

Provides a drop-in replacement for :class:`nequix.nequix.Nequix` that implements
the MACE (Multi-Atomic Cluster Expansion) architecture. The ``__call__``
signature matches ``Nequix.__call__`` so it can be swapped in directly.

The port targets a minimal but complete MACE:
``RealAgnosticResidualInteractionBlock`` + ``EquivariantProductBasisBlock`` +
``ScaleShift``, without cuequivariance, LAMMPS, multi-head support, ZBL
pair repulsion, or distance transforms.
"""

import json
import math
from typing import Callable, Optional, Sequence

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph

from nequix.nequix import (
    MLP,
    bessel_basis,
    node_graph_idx,
    polynomial_cutoff,
)


class SymmetricContraction(eqx.Module):
    """Symmetric contraction block (MACE body-order expansion).

    Computes ``out = sum_{nu=1..correlation} f_nu(x, ..., x)`` where each
    ``f_nu`` is a symmetric nu-fold tensor contraction with element-dependent
    learnable mixing weights. The Clebsch-Gordan bases are precomputed via
    :func:`e3nn.reduced_symmetric_tensor_product_basis` and treated as
    constants at training time.

    The input and output irreps must share a single common multiplicity (the
    MACE "channel" convention).
    """

    irreps_in: e3nn.Irreps = eqx.field(static=True)
    irreps_out: e3nn.Irreps = eqx.field(static=True)
    correlation: int = eqx.field(static=True)
    num_elements: int = eqx.field(static=True)
    mul: int = eqx.field(static=True)
    in_slices: tuple = eqx.field(static=True)
    out_slices: tuple = eqx.field(static=True)
    entries: tuple = eqx.field(static=True)

    bases: list
    weights: list

    def __init__(
        self,
        key: jax.Array,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        correlation: int,
        num_elements: int,
    ):
        self.irreps_in = e3nn.Irreps(irreps_in).simplify()
        self.irreps_out = e3nn.Irreps(irreps_out).simplify()
        self.correlation = correlation
        self.num_elements = num_elements

        muls = {mul for mul, _ in self.irreps_in} | {mul for mul, _ in self.irreps_out}
        if len(muls) != 1:
            raise ValueError(
                "SymmetricContraction requires a single shared multiplicity "
                f"across irreps_in and irreps_out, got {self.irreps_in} and "
                f"{self.irreps_out}"
            )
        self.mul = muls.pop()

        # Per-irrep block offsets in the per-mul (n_nodes, mul, dim_single) layout
        in_slices = []
        offset = 0
        for _, ir in self.irreps_in:
            in_slices.append((offset, offset + ir.dim))
            offset += ir.dim
        self.in_slices = tuple(in_slices)

        out_slices = []
        offset = 0
        for _, ir in self.irreps_out:
            out_slices.append((offset, offset + ir.dim))
            offset += ir.dim
        self.out_slices = tuple(out_slices)

        irreps_in_single = e3nn.Irreps([(1, ir) for _, ir in self.irreps_in])
        d_in_single = irreps_in_single.dim

        bases: list[jax.Array] = []
        weights: list[jax.Array] = []
        entries: list[tuple[int, int, int]] = []

        max_entries = correlation * len(self.irreps_out)
        keys = jax.random.split(key, max(max_entries, 1))

        for nu in range(1, correlation + 1):
            for out_idx, (_, ir_out) in enumerate(self.irreps_out):
                basis_ia = e3nn.reduced_symmetric_tensor_product_basis(
                    irreps_in_single, nu, keep_ir=[ir_out]
                )
                n_basis = basis_ia.irreps.count(ir_out)
                if n_basis == 0:
                    continue

                # Reshape flattened last dim into (n_basis, ir_out.dim)
                basis_arr = basis_ia.array.reshape(
                    (d_in_single,) * nu + (n_basis, ir_out.dim)
                )
                bases.append(basis_arr.astype(jnp.float32))

                k = keys[len(entries)]
                scale = 1.0 / math.sqrt(n_basis)
                w = (
                    jax.random.normal(k, (num_elements, n_basis, self.mul))
                    * scale
                )
                weights.append(w)

                entries.append((nu, out_idx, n_basis))

        self.bases = bases
        self.weights = weights
        self.entries = tuple(entries)

    def _to_per_mul(self, x: jax.Array) -> jax.Array:
        """Convert a flat (n_nodes, irreps.dim) array to (n_nodes, mul, dim_single)."""
        pieces = []
        n_nodes = x.shape[0]
        offset = 0
        for mul, ir in self.irreps_in:
            size = mul * ir.dim
            piece = x[:, offset : offset + size].reshape(n_nodes, mul, ir.dim)
            pieces.append(piece)
            offset += size
        return jnp.concatenate(pieces, axis=-1)

    def _from_per_mul(self, x: jax.Array) -> jax.Array:
        """Convert (n_nodes, mul, dim_single_out) back to flat irreps layout."""
        n_nodes = x.shape[0]
        pieces = []
        for out_idx, (mul, ir) in enumerate(self.irreps_out):
            lo, hi = self.out_slices[out_idx]
            piece = x[:, :, lo:hi].reshape(n_nodes, mul * ir.dim)
            pieces.append(piece)
        return jnp.concatenate(pieces, axis=-1)

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        species: jax.Array,
    ) -> e3nn.IrrepsArray:
        feats = self._to_per_mul(node_feats.array)
        n_nodes = feats.shape[0]

        out = jnp.zeros(
            (n_nodes, self.mul, self.irreps_out.dim // self.mul),
            dtype=feats.dtype,
        )

        for i, (nu, out_idx, _) in enumerate(self.entries):
            basis = jax.lax.stop_gradient(self.bases[i]).astype(feats.dtype)
            # basis shape: (d_in,)*nu + (n_basis, d_out)

            # First contraction introduces the (n_nodes, mul) axes.
            tmp = jnp.einsum("a...,ima->im...", basis, feats)
            # tmp shape: (n_nodes, mul, (d_in,)*(nu-1), n_basis, d_out)

            # Subsequent contractions fold over (n_nodes, mul) as batch dims.
            for _ in range(nu - 1):
                tmp = jnp.einsum("ima...,ima->im...", tmp, feats)
            # tmp shape: (n_nodes, mul, n_basis, d_out)

            w_per_node = self.weights[i][species]  # (n_nodes, n_basis, mul)
            contracted = jnp.einsum("nsm,nmsp->nmp", w_per_node, tmp)

            lo, hi = self.out_slices[out_idx]
            out = out.at[:, :, lo:hi].add(contracted)

        return e3nn.IrrepsArray(self.irreps_out, self._from_per_mul(out))


class MACEInteractionBlock(eqx.Module):
    """MACE interaction block matching ``RealAgnosticResidualInteractionBlock``.

    Computes an equivariant message via a weighted tensor product between
    sender features and edge spherical harmonics, scatter-aggregates into
    receiver nodes, and emits a species-indexed skip connection to be mixed
    with the symmetric contraction output downstream.
    """

    target_irreps: e3nn.Irreps = eqx.field(static=True)
    hidden_irreps: e3nn.Irreps = eqx.field(static=True)
    avg_num_neighbors: float = eqx.field(static=True)

    linear_up: e3nn.equinox.Linear
    radial_mlp: MLP
    linear: e3nn.equinox.Linear
    skip_tp: e3nn.equinox.Linear

    def __init__(
        self,
        key: jax.Array,
        node_feats_irreps: e3nn.Irreps,
        edge_attrs_irreps: e3nn.Irreps,
        target_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_num_neighbors: float,
    ):
        node_feats_irreps = e3nn.Irreps(node_feats_irreps)
        edge_attrs_irreps = e3nn.Irreps(edge_attrs_irreps)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.hidden_irreps = e3nn.Irreps(hidden_irreps)
        self.avg_num_neighbors = avg_num_neighbors

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_up = e3nn.equinox.Linear(
            irreps_in=node_feats_irreps,
            irreps_out=node_feats_irreps,
            key=k1,
        )

        tp_irreps = e3nn.tensor_product(
            node_feats_irreps, edge_attrs_irreps, filter_ir_out=self.target_irreps
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            key=k2,
        )

        self.linear = e3nn.equinox.Linear(
            irreps_in=tp_irreps,
            irreps_out=self.target_irreps,
            key=k3,
            force_irreps_out=True,
        )

        # Species-indexed skip: emulates FullyConnectedTensorProduct
        # (node_feats, one_hot_species) -> hidden_irreps.
        self.skip_tp = e3nn.equinox.Linear(
            irreps_in=node_feats_irreps,
            irreps_out=self.hidden_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            key=k4,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        sc = self.skip_tp(species, node_feats)

        features = self.linear_up(node_feats)
        messages = features[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.target_irreps)

        radial_weights = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_weights

        message_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=node_feats.shape[0]
        )
        message = self.linear(message_agg) / jax.lax.stop_gradient(
            jnp.asarray(self.avg_num_neighbors)
        )

        return message, sc


class EquivariantProductBasisBlock(eqx.Module):
    """Symmetric contraction followed by a linear mix and a skip connection."""

    use_sc: bool = eqx.field(static=True)
    symmetric_contraction: SymmetricContraction
    linear: e3nn.equinox.Linear

    def __init__(
        self,
        key: jax.Array,
        node_feats_irreps: e3nn.Irreps,
        target_irreps: e3nn.Irreps,
        correlation: int,
        num_elements: int,
        use_sc: bool = True,
    ):
        self.use_sc = use_sc
        k1, k2 = jax.random.split(key)
        self.symmetric_contraction = SymmetricContraction(
            key=k1,
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        self.linear = e3nn.equinox.Linear(
            irreps_in=target_irreps,
            irreps_out=target_irreps,
            key=k2,
            force_irreps_out=True,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        sc: Optional[e3nn.IrrepsArray],
        species: jax.Array,
    ) -> e3nn.IrrepsArray:
        x = self.symmetric_contraction(node_feats, species)
        x = self.linear(x)
        if self.use_sc and sc is not None:
            x = x + sc
        return x


class LinearReadout(eqx.Module):
    """Linear readout producing one scalar per node."""

    linear: e3nn.equinox.Linear

    def __init__(self, key: jax.Array, irreps_in: e3nn.Irreps):
        self.linear = e3nn.equinox.Linear(
            irreps_in=e3nn.Irreps(irreps_in),
            irreps_out="0e",
            key=key,
            force_irreps_out=True,
        )

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return self.linear(x)


class NonLinearReadout(eqx.Module):
    """Two-layer readout with a scalar activation in between."""

    activation: Callable = eqx.field(static=True)
    linear_1: e3nn.equinox.Linear
    linear_2: e3nn.equinox.Linear

    def __init__(
        self,
        key: jax.Array,
        irreps_in: e3nn.Irreps,
        mlp_irreps: e3nn.Irreps,
        activation: Callable = jax.nn.silu,
    ):
        self.activation = activation
        mlp_irreps = e3nn.Irreps(mlp_irreps).filter("0e")
        if mlp_irreps.dim == 0:
            raise ValueError(
                "NonLinearReadout requires at least one scalar (0e) in mlp_irreps"
            )
        k1, k2 = jax.random.split(key)
        self.linear_1 = e3nn.equinox.Linear(
            irreps_in=e3nn.Irreps(irreps_in),
            irreps_out=mlp_irreps,
            key=k1,
            force_irreps_out=True,
        )
        self.linear_2 = e3nn.equinox.Linear(
            irreps_in=mlp_irreps,
            irreps_out="0e",
            key=k2,
            force_irreps_out=True,
        )

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        x = self.linear_1(x)
        x = e3nn.IrrepsArray(x.irreps, self.activation(x.array))
        return self.linear_2(x)


class MACE(eqx.Module):
    """MACE model with a call signature matching :class:`nequix.nequix.Nequix`.

    Returns ``(graph_energies, forces, stress)`` from a
    :class:`jraph.GraphsTuple`, where ``stress`` is ``None`` for non-periodic
    inputs.
    """

    n_species: int = eqx.field(static=True)
    max_ell: int = eqx.field(static=True)
    spatial_cutoff: float = eqx.field(static=True)
    radial_basis_size: int = eqx.field(static=True)
    radial_polynomial_p: float = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    atom_energies: jax.Array
    node_embedding: e3nn.equinox.Linear
    interactions: list
    products: list
    readouts: list

    def __init__(
        self,
        key: jax.Array,
        n_species: int,
        max_ell: int = 3,
        spatial_cutoff: float = 5.0,
        hidden_irreps: str = "128x0e + 128x1o + 128x2e",
        num_interactions: int = 2,
        correlation: int = 3,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 6.0,
        mlp_init_scale: float = 4.0,
        MLP_irreps: str = "16x0e",
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
    ):
        self.n_species = n_species
        self.max_ell = max_ell
        self.spatial_cutoff = spatial_cutoff
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.shift = shift
        self.scale = scale
        self.atom_energies = (
            jnp.array(atom_energies, dtype=jnp.float32)
            if atom_energies is not None
            else jnp.zeros(n_species, dtype=jnp.float32)
        )

        hidden_irreps = e3nn.Irreps(hidden_irreps)
        mlp_irreps = e3nn.Irreps(MLP_irreps)
        sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)

        num_features = hidden_irreps.count(e3nn.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

        node_attr_irreps = e3nn.Irreps(f"{n_species}x0e")
        node_feats_irreps = e3nn.Irreps(f"{num_features}x0e")

        keys = jax.random.split(key, 1 + 3 * num_interactions)
        key_iter = iter(keys)

        self.node_embedding = e3nn.equinox.Linear(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            key=next(key_iter),
            force_irreps_out=True,
        )

        interactions: list = []
        products: list = []
        readouts: list = []

        for i in range(num_interactions):
            is_last = i == num_interactions - 1

            # Last layer collapses to the scalar portion of hidden_irreps only.
            if is_last:
                hidden_irreps_out = e3nn.Irreps(str(hidden_irreps[0]))
            else:
                hidden_irreps_out = hidden_irreps

            ibl_in_irreps = node_feats_irreps if i == 0 else hidden_irreps

            interactions.append(
                MACEInteractionBlock(
                    key=next(key_iter),
                    node_feats_irreps=ibl_in_irreps,
                    edge_attrs_irreps=sh_irreps,
                    target_irreps=interaction_irreps,
                    hidden_irreps=hidden_irreps_out,
                    n_species=n_species,
                    radial_basis_size=radial_basis_size,
                    radial_mlp_size=radial_mlp_size,
                    radial_mlp_layers=radial_mlp_layers,
                    mlp_init_scale=mlp_init_scale,
                    avg_num_neighbors=avg_n_neighbors,
                )
            )

            products.append(
                EquivariantProductBasisBlock(
                    key=next(key_iter),
                    node_feats_irreps=interaction_irreps,
                    target_irreps=hidden_irreps_out,
                    correlation=correlation,
                    num_elements=n_species,
                    use_sc=True,
                )
            )

            if is_last:
                readouts.append(
                    NonLinearReadout(
                        key=next(key_iter),
                        irreps_in=hidden_irreps_out,
                        mlp_irreps=mlp_irreps,
                    )
                )
            else:
                readouts.append(
                    LinearReadout(
                        key=next(key_iter),
                        irreps_in=hidden_irreps_out,
                    )
                )

        self.interactions = interactions
        self.products = products
        self.readouts = readouts

    def node_energies(
        self,
        displacements: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        node_attrs = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"),
            jax.nn.one_hot(species, self.n_species),
        )

        # Safe norm (avoids nan for r = 0 on padded edges)
        square_r_norm = jnp.sum(displacements**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        cutoff = polynomial_cutoff(
            r_norm, self.spatial_cutoff, self.radial_polynomial_p
        )
        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, self.spatial_cutoff)
            * cutoff[:, None]
        )

        sh = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.max_ell),
            displacements,
            normalize=True,
            normalization="component",
        )

        node_feats = self.node_embedding(node_attrs)

        n_nodes = species.shape[0]
        node_inter_es = jnp.zeros((n_nodes, 1), dtype=node_feats.array.dtype)

        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            message, sc = interaction(
                node_feats=node_feats,
                species=species,
                sh=sh,
                radial_basis=radial_basis,
                senders=senders,
                receivers=receivers,
            )
            node_feats = product(message, sc, species)
            node_inter_es = node_inter_es + readout(node_feats).array

        # ScaleShiftMACE: scale/shift applied to interaction energies only,
        # then added to e0 (atom_energies) with stop_gradient.
        node_inter_es = node_inter_es * jax.lax.stop_gradient(
            jnp.asarray(self.scale)
        ) + jax.lax.stop_gradient(jnp.asarray(self.shift))
        node_energies = node_inter_es + jax.lax.stop_gradient(
            self.atom_energies[species, None]
        )

        return node_energies

    def __call__(self, data: jraph.GraphsTuple):
        batch_index = node_graph_idx(data)

        if data.globals["cell"] is None:
            def total_energy_fn(positions: jax.Array):
                r = positions[data.senders] - positions[data.receivers]
                node_energies = self.node_energies(
                    r,
                    data.nodes["species"],
                    data.senders,
                    data.receivers,
                )
                return jnp.sum(node_energies), node_energies

            minus_forces, node_energies = eqx.filter_grad(
                total_energy_fn, has_aux=True
            )(data.nodes["positions"])
        else:
            def total_energy_fn(positions_eps: tuple[jax.Array, jax.Array]):
                positions, eps = positions_eps
                eps_sym = (eps + eps.swapaxes(1, 2)) / 2
                eps_sym_per_node = jnp.repeat(
                    eps_sym,
                    data.n_node,
                    axis=0,
                    total_repeat_length=data.nodes["positions"].shape[0],
                )
                positions = positions + jnp.einsum(
                    "ik,ikj->ij", positions, eps_sym_per_node
                )
                cell = data.globals["cell"] + jnp.einsum(
                    "bij,bjk->bik", data.globals["cell"], eps_sym
                )
                cell_per_edge = jnp.repeat(
                    cell,
                    data.n_edge,
                    axis=0,
                    total_repeat_length=data.edges["shifts"].shape[0],
                )
                offsets = jnp.einsum(
                    "ij,ijk->ik", data.edges["shifts"], cell_per_edge
                )
                r = positions[data.senders] - positions[data.receivers] + offsets
                node_energies = self.node_energies(
                    r,
                    data.nodes["species"],
                    data.senders,
                    data.receivers,
                )
                return jnp.sum(node_energies), node_energies

            eps = jnp.zeros_like(data.globals["cell"])
            (minus_forces, virial), node_energies = eqx.filter_grad(
                total_energy_fn, has_aux=True
            )((data.nodes["positions"], eps))

        node_mask = jraph.get_node_padding_mask(data)
        minus_forces = jnp.where(node_mask[:, None], minus_forces, 0.0)

        graph_energies = jraph.segment_sum(
            node_energies,
            batch_index,
            num_segments=data.n_node.shape[0],
            indices_are_sorted=True,
        )

        if data.globals["cell"] is None:
            stress = None
        else:
            det = jnp.abs(jnp.linalg.det(data.globals["cell"]))[:, None, None]
            det = jnp.where(det > 0.0, det, 1.0)
            stress = virial / det
            graph_mask = jraph.get_graph_padding_mask(data)
            stress = jnp.where(graph_mask[:, None, None], stress, 0.0)

        return graph_energies[:, 0], -minus_forces, stress


def get_param_labels(model_tree):
    """Assign weight decay labels strictly based on parameter paths."""

    def label_fn(path, leaf):
        if not eqx.is_inexact_array(leaf):
            return "ignore"

        path_str = "".join(str(p) for p in path)

        # Biases, layer norms, atom energies, and CG basis constants are never decayed.
        if (
            "bias" in path_str
            or "layer_norm" in path_str
            or "atom_energies" in path_str
            or "bases" in path_str
        ):
            return "ignore"

        return "decay_regular"

    return jax.tree_util.tree_map_with_path(label_fn, model_tree)


def save_model(path: str, model: eqx.Module, config: dict):
    """Save a MACE model and its config to a file."""
    with open(path, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str) -> tuple[MACE, dict]:
    """Load a MACE model and its config from a file."""
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        model = MACE(
            key=jax.random.key(0),
            n_species=len(config["atomic_numbers"]),
            max_ell=config["max_ell"],
            spatial_cutoff=config.get("spatial_cutoff", config.get("cutoff")),
            hidden_irreps=config["hidden_irreps"],
            num_interactions=config["num_interactions"],
            correlation=config["correlation"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
            MLP_irreps=config.get("MLP_irreps", "16x0e"),
            shift=config["shift"],
            scale=config["scale"],
            avg_n_neighbors=config["avg_n_neighbors"],
            # NOTE: atom_energies are recovered from serialized weights
        )
        model = eqx.tree_deserialise_leaves(f, model)
        return model, config
