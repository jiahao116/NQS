"""quantax operator for the half-filled mixD multilayer Heisenberg model.

Built from the same bond list as the ED backends (lattice.HeisenbergCouplings),
in the spin-1/2 S.S convention:
    S_i.S_j = 1/4 sz_i sz_j + 1/2 (s+_i s-_j + s-_i s+_j)   (sigma-matrix primitives)
With msr=True the off-diagonal sign is flipped on every bond (all bonds connect
opposite sublattices on a bipartite multilayer cluster), which applies the Marshall
sign rule as a basis rotation; energies are unchanged.

Status: written against quantax 0.4.0 primitives (sigma_p, sigma_m, sigma_z) but not
yet executed (quantax is not installed in the local environment). Validate with
``quantax`` exact diagonalization against ed_numba before any NQS run.
"""

from __future__ import annotations


def mixd_heisenberg_operator(bond_list, msr: bool = True):
    from quantax.operator import Operator, sigma_m, sigma_p, sigma_z

    sign = -1.0 if msr else 1.0
    terms = []
    for i, j, J, _label in bond_list:
        terms.append(0.25 * J * (sigma_z(i) @ sigma_z(j)))
        terms.append(0.5 * sign * J * (sigma_p(i) @ sigma_m(j) + sigma_m(i) @ sigma_p(j)))
    return sum(terms, Operator([]))


def make_quantax_lattice(lat):
    """quantax Grid with the same site order as MultilayerLattice (layer = last axis)."""
    import quantax as qtx

    boundary = [1 if lat.pbc_x else 0, 1 if lat.pbc_y else 0, 0]
    return qtx.sites.Grid([lat.lx, lat.ly, lat.n_layers], boundary=boundary)
