"""Geometry of mixed-dimensional (mixD) multilayer square lattices.

Sites live on an Lx x Ly x n_layers grid. In-plane bonds connect nearest neighbours
within a layer (periodic or open per in-plane direction); interlayer bonds connect
vertically adjacent sites (layer l to l+1, always open, i.e. no l=n-1 -> 0 bond).

Bond lists are deduplicated: for a periodic direction of length 2 the +x and -x
neighbours coincide and the pair is counted once. Every consumer (numba ED, QuSpin
ED, quantax operator) builds its Hamiltonian from these lists, so all backends share
exactly the same bond convention by construction.

Site index convention (matches quantax Grid row-major order for extent
[Lx, Ly, n_layers]): idx = (x * Ly + y) * n_layers + layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MultilayerLattice:
    lx: int
    ly: int
    n_layers: int
    pbc_x: bool = True
    pbc_y: bool = True

    @property
    def n_sites(self) -> int:
        return self.lx * self.ly * self.n_layers

    @property
    def n_columns(self) -> int:
        return self.lx * self.ly

    def index(self, x: int, y: int, layer: int) -> int:
        return (x * self.ly + y) * self.n_layers + layer

    def coords(self, idx: int) -> tuple[int, int, int]:
        layer = idx % self.n_layers
        col = idx // self.n_layers
        return col // self.ly, col % self.ly, layer

    def _inplane_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        out = []
        if self.pbc_x or x + 1 < self.lx:
            out.append(((x + 1) % self.lx, y))
        if self.pbc_y or y + 1 < self.ly:
            out.append((x, (y + 1) % self.ly))
        return out

    def inplane_bonds(self) -> list[tuple[int, int, int]]:
        """Unique in-plane bonds as (i, j, layer) with i < j."""
        seen: set[tuple[int, int]] = set()
        bonds = []
        for layer in range(self.n_layers):
            for x in range(self.lx):
                for y in range(self.ly):
                    i = self.index(x, y, layer)
                    for xn, yn in self._inplane_neighbors(x, y):
                        j = self.index(xn, yn, layer)
                        if i == j:
                            continue
                        key = (min(i, j), max(i, j))
                        if key in seen:
                            continue
                        seen.add(key)
                        bonds.append((key[0], key[1], layer))
        return bonds

    def interlayer_bonds(self) -> list[tuple[int, int, int]]:
        """Unique interlayer bonds as (i, j, lower_layer) with j directly above i."""
        bonds = []
        for x in range(self.lx):
            for y in range(self.ly):
                for layer in range(self.n_layers - 1):
                    bonds.append(
                        (self.index(x, y, layer), self.index(x, y, layer + 1), layer)
                    )
        return bonds

    def sublattice_sign(self) -> np.ndarray:
        """(-1)^(x+y+layer); the lattice is bipartite iff every periodic length is even."""
        signs = np.empty(self.n_sites, dtype=np.int64)
        for idx in range(self.n_sites):
            x, y, layer = self.coords(idx)
            signs[idx] = 1 if (x + y + layer) % 2 == 0 else -1
        return signs

    def is_bipartite(self) -> bool:
        ok_x = (not self.pbc_x) or self.lx % 2 == 0 or self.lx == 1
        ok_y = (not self.pbc_y) or self.ly % 2 == 0 or self.ly == 1
        return ok_x and ok_y

    def label(self) -> str:
        bc = ("p" if self.pbc_x else "o") + ("p" if self.pbc_y else "o")
        return f"{self.lx}x{self.ly}x{self.n_layers}_{bc}"


@dataclass
class HeisenbergCouplings:
    """Half-filled mixD limit: H = J_par sum_inplane S.S + J_perp sum_inter S.S.

    Spin-1/2 operators S (not Pauli sigma). quantax.operator.Heisenberg uses
    sigma.sigma = 4 S.S, so its energies are 4x the ones defined here.
    The -n_i n_j / 4 term of the t-J model is a constant at half filling and is
    omitted here; see ``tj_constant_shift`` to convert.
    """

    j_par: float = 1.0
    j_perp: float = 1.0
    bond_list: list[tuple[int, int, float, str]] = field(default_factory=list)

    @classmethod
    def build(cls, lat: MultilayerLattice, j_par: float, j_perp: float):
        bonds = [(i, j, j_par, f"par_L{layer}") for i, j, layer in lat.inplane_bonds()]
        bonds += [
            (i, j, j_perp, f"perp_L{layer}{layer + 1}")
            for i, j, layer in lat.interlayer_bonds()
        ]
        return cls(j_par=j_par, j_perp=j_perp, bond_list=bonds)

    def tj_constant_shift(self) -> float:
        """Energy offset -sum_b J_b/4 that the t-J form adds at half filling."""
        return -0.25 * sum(jb for _, _, jb, _ in self.bond_list)
