"""QuSpin exact diagonalization for the half-filled mixD multilayer Heisenberg model.

Second, independent ED path (cross-check of ed_numba). Uses spin-1/2 operators
(pauli=0) so energies are in the S.S convention of lattice.HeisenbergCouplings.
Requires: pip install quspin (not available in the base environment yet).
"""

from __future__ import annotations

import numpy as np


def quspin_heisenberg_ed(
    n_sites: int,
    bonds: list[tuple[int, int, float]],
    n_up: int | None = None,
    k: int = 1,
    return_bond_corr: bool = True,
):
    from quspin.basis import spin_basis_general
    from quspin.operators import hamiltonian

    if n_up is None:
        n_up = n_sites // 2
    basis = spin_basis_general(n_sites, Nup=n_up, pauli=0)
    zz = [[J, i, j] for i, j, J in bonds]
    pm = [[0.5 * J, i, j] for i, j, J in bonds]
    static = [["zz", zz], ["+-", pm], ["-+", pm]]
    no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)
    H = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
    if basis.Ns <= 2000:
        w, v = np.linalg.eigh(H.toarray())
        w, v = w[:k], v[:, :k]
    else:
        w, v = H.eigsh(k=k, which="SA")
        order = np.argsort(w)
        w, v = w[order], v[:, order]
    psi0 = v[:, 0]

    out = {"energies": np.asarray(w), "dim": int(basis.Ns)}
    if return_bond_corr:
        corr = []
        for i, j, _ in bonds:
            op = hamiltonian(
                [["zz", [[1.0, i, j]]], ["+-", [[0.5, i, j]]], ["-+", [[0.5, i, j]]]],
                [], basis=basis, dtype=np.float64, **no_checks,
            )
            corr.append(float(op.expt_value(psi0).real))
        out["bond_corr"] = np.array(corr)
    return out
