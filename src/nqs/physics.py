"""Quantum spin Hamiltonians and Exact Diagonalization (ED) reference solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nqs.config import SystemConfig


@dataclass
class QuantumSystem:
    """Container for NetKet Hilbert space, lattice graph, Hamiltonian, and exact benchmarks."""

    hilbert: Any
    graph: Any
    hamiltonian: Any
    exact_ground_energy: float | None = None
    exact_first_excited: float | None = None

    @property
    def exact_gap(self) -> float | None:
        if self.exact_ground_energy is not None and self.exact_first_excited is not None:
            return float(self.exact_first_excited - self.exact_ground_energy)
        return None


def build_quantum_system(cfg: SystemConfig) -> QuantumSystem:
    """Construct the NetKet spin system and optionally solve Exact Diagonalization via Lanczos."""
    import netket as nk

    if cfg.hamiltonian == "tfim_1d":
        graph = nk.graph.Chain(length=cfg.n_spins, pbc=cfg.periodic)
        # Unconstrained Spin-1/2 space so autoregressive direct samplers (ARDirectSampler) work natively
        hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
        hamiltonian = nk.operator.Ising(
            hilbert=hilbert,
            graph=graph,
            h=cfg.h_field,
            J=cfg.j_coupling,
        )
    elif cfg.hamiltonian == "heisenberg_1d":
        graph = nk.graph.Chain(length=cfg.n_spins, pbc=cfg.periodic)
        hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
        hamiltonian = nk.operator.Heisenberg(
            hilbert=hilbert,
            graph=graph,
            J=cfg.j_coupling,
        )
    elif cfg.hamiltonian == "heisenberg_2d":
        graph = nk.graph.Grid(extent=[cfg.lx, cfg.ly], pbc=cfg.periodic)
        hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
        hamiltonian = nk.operator.Heisenberg(
            hilbert=hilbert,
            graph=graph,
            J=cfg.j_coupling,
        )
    else:
        raise ValueError(f"Unsupported hamiltonian: {cfg.hamiltonian}")

    exact_e0: float | None = None
    exact_e1: float | None = None
    if cfg.compute_exact:
        evals = nk.exact.lanczos_ed(hamiltonian, k=2, compute_eigenvectors=False)
        exact_e0 = float(evals[0])
        exact_e1 = float(evals[1]) if len(evals) > 1 else None

    return QuantumSystem(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        exact_ground_energy=exact_e0,
        exact_first_excited=exact_e1,
    )
