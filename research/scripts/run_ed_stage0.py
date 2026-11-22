"""Stage 0: exact diagonalization of the half-filled mixD multilayer Heisenberg model.

Usage (from research/):
    python scripts/run_ed_stage0.py [--config configs/stage0_ed_scan.yaml] [--max-sites 24]

Writes one JSON per (cluster, J_par, J_perp) to <output_dir>/ and a summary.md with
analytic-limit checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mixd.ed_numba import heisenberg_ed  # noqa: E402
from mixd.lattice import HeisenbergCouplings, MultilayerLattice  # noqa: E402

# 4x4 periodic square lattice, S.S convention. Literature value E0/N = -0.70178020
# (e.g. Schulz, Ziman, Poilblanc 1996). Note: the public repo README quotes the QuSpin
# value -44.91392517 (sigma.sigma), i.e. -11.22848129 here, which is 1.9e-6 too high.
E_4X4_REF = -0.70178020 * 16
E_4X4_REPO = -44.91392517 / 4.0


def tag(lat: MultilayerLattice, j_par: float, j_perp: float) -> str:
    return f"{lat.label()}_jpar{j_par:g}_jperp{j_perp:g}"


def observables(lat, cpl, C):
    eps = lat.sublattice_sign().astype(float)
    n = lat.n_sites
    classes: dict[str, list[float]] = {}
    e_from_corr = 0.0
    for i, j, jb, label in cpl.bond_list:
        classes.setdefault(label, []).append(C[i, j])
        e_from_corr += jb * C[i, j]
    s_tot2 = float(C.sum())
    s_pi = float(eps @ C @ eps / n)
    layer_m2 = {}
    for layer in range(lat.n_layers):
        idx = [k for k in range(n) if lat.coords(k)[2] == layer]
        e = eps[idx]
        layer_m2[f"L{layer}"] = float(e @ C[np.ix_(idx, idx)] @ e / len(idx) ** 2)
    return {
        "bond_corr_mean": {k: float(np.mean(v)) for k, v in sorted(classes.items())},
        "energy_from_correlations": float(e_from_corr),
        "S_tot_squared": s_tot2,
        "S_pi": s_pi,
        "m_stag_squared": s_pi / n,
        "m_stag_squared_per_layer": layer_m2,
    }


def run_point(lat, j_par, j_perp, spin_gap, correlations):
    cpl = HeisenbergCouplings.build(lat, j_par, j_perp)
    bonds = [(i, j, jb) for i, j, jb, _ in cpl.bond_list]
    t0 = time.time()
    res = heisenberg_ed(lat.n_sites, bonds, with_correlations=correlations)
    out = {
        "cluster": lat.label(),
        "lx": lat.lx, "ly": lat.ly, "n_layers": lat.n_layers,
        "n_sites": lat.n_sites, "n_bonds": len(bonds),
        "j_par": j_par, "j_perp": j_perp,
        "convention": "spin-1/2 S.S; quantax Heisenberg (sigma.sigma) energy = 4x",
        "dim_sz0": res.dim,
        "E0": float(res.energies[0]),
        "E0_per_site": float(res.energies[0] / lat.n_sites),
        "E0_tJ_convention": float(res.energies[0] + cpl.tj_constant_shift()),
    }
    if correlations:
        out.update(observables(lat, cpl, res.correlations))
    if spin_gap:
        r1 = heisenberg_ed(lat.n_sites, bonds, n_up=lat.n_sites // 2 + 1)
        out["E0_sz1"] = float(r1.energies[0])
        out["spin_gap"] = float(r1.energies[0] - res.energies[0])
    out["wall_time_s"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "stage0_ed_scan.yaml"))
    ap.add_argument("--max-sites", type=int, default=24)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    lattices = [MultilayerLattice(**c) for c in cfg["clusters"]]
    lattices = [lat for lat in lattices if lat.n_sites <= args.max_sites]
    lattices.sort(key=lambda lat: lat.n_sites)
    results = {}
    for lat in lattices:
        assert lat.is_bipartite(), lat.label()
        points = [(cfg["j_par"], jp) for jp in cfg["j_perp_values"]]
        if lat.n_layers == 1:
            points = [(cfg["j_par"], 0.0)]
        elif cfg.get("decoupled_columns_check", False):
            points.append((0.0, 1.0))
        for j_par, j_perp in points:
            name = tag(lat, j_par, j_perp)
            print(f"[ed] {name} ...", flush=True)
            r = run_point(lat, j_par, j_perp, cfg.get("spin_gap", True), cfg.get("correlations", True))
            results[name] = r
            (out_dir / f"{name}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(
                f"[ed] {name}: dim={r['dim_sz0']} E0={r['E0']:.10f} "
                f"E0/N={r['E0_per_site']:.10f} gap={r.get('spin_gap', float('nan')):.6f} "
                f"t={r['wall_time_s']}s",
                flush=True,
            )

    checks = analytic_checks(results)
    write_summary(out_dir, results, checks)
    print("\n".join(f"[check] {'PASS' if ok else 'FAIL'} {msg}" for ok, msg in checks))
    return 0 if all(ok for ok, _ in checks) else 1


def analytic_checks(results):
    checks = []
    tol = 1e-7

    def get(name):
        return results.get(name)

    r = get("4x4x1_pp_jpar1_jperp0")
    if r:
        d = r["E0"] - E_4X4_REF
        checks.append((abs(d) < 1e-6, f"4x4 single layer E0={r['E0']:.8f} vs literature {E_4X4_REF:.8f} (diff {d:.1e}); repo README value {E_4X4_REPO:.8f} differs by {r['E0'] - E_4X4_REPO:.1e}"))
        r2 = get("4x2x2_pp_jpar1_jperp1")
        if r2:
            d = r2["E0"] - r["E0"]
            checks.append((abs(d) < tol, f"graph isomorphism: 4x2x2 (J_perp=J_par, Ly=2 deduplicated) == 4x4 torus (diff {d:.1e})"))
    base = get("4x2x1_pp_jpar1_jperp0")
    for nl in (2, 3):
        r = get(f"4x2x{nl}_pp_jpar1_jperp0")
        if r and base:
            d = r["E0"] - nl * base["E0"]
            checks.append((abs(d) < tol, f"J_perp=0: E(4x2x{nl}) = {nl} E(4x2x1) (diff {d:.1e})"))
        r = get(f"4x2x{nl}_pp_jpar0_jperp1")
        if r:
            exact = -0.75 * 8 if nl == 2 else -1.0 * 8
            d = r["E0"] - exact
            checks.append((abs(d) < tol, f"J_par=0: {nl}-site columns E0={r['E0']:.8f} vs analytic {exact} (diff {d:.1e})"))
    for name, r in results.items():
        if "energy_from_correlations" in r:
            d = r["energy_from_correlations"] - r["E0"]
            checks.append((abs(d) < 1e-7, f"{name}: sum_b J_b <S.S>_b = E0 (diff {d:.1e})"))
        degenerate = r["n_layers"] == 3 and r["j_par"] == 0.0
        if "S_tot_squared" in r and not degenerate:
            checks.append((abs(r["S_tot_squared"]) < 1e-6, f"{name}: singlet ground state, <S_tot^2>={r['S_tot_squared']:.1e}"))
    return checks


def write_summary(out_dir, results, checks):
    lines = [
        "# Stage 0 ED: half-filled mixD multilayer Heisenberg",
        "",
        "Convention: H = J_par sum_inplane S.S + J_perp sum_interlayer S.S (spin-1/2). "
        "Clusters periodic in-plane (bonds deduplicated for length-2 directions), open between layers. "
        "Backend: matrix-free numba ED (research/src/mixd/ed_numba.py).",
        "",
        "| cluster | J_perp/J_par | dim(Sz=0) | E0 | E0/N | spin gap | <S.S> in-plane (per layer) | <S.S> interlayer | m_stag^2 |",
        "| :-- | --: | --: | --: | --: | --: | :-- | :-- | --: |",
    ]
    for name, r in results.items():
        bc = r.get("bond_corr_mean", {})
        par = ", ".join(f"{k[4:]}:{v:.4f}" for k, v in bc.items() if k.startswith("par"))
        perp = ", ".join(f"{k[5:]}:{v:.4f}" for k, v in bc.items() if k.startswith("perp"))
        ratio = "inf" if r["j_par"] == 0 else f"{r['j_perp'] / r['j_par']:g}"
        lines.append(
            f"| {r['cluster']} | {ratio} | {r['dim_sz0']} | {r['E0']:.8f} | {r['E0_per_site']:.8f} | "
            f"{r.get('spin_gap', float('nan')):.5f} | {par} | {perp or '-'} | {r.get('m_stag_squared', float('nan')):.5f} |"
        )
    lines += ["", "## Analytic-limit checks", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {msg}" for ok, msg in checks]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
