"""quantax NQS runner for the half-filled mixD multilayer Heisenberg model (Stage 0).

Pipeline:
  1. Build the multilayer Grid and the operator from the shared bond list.
  2. Operator check: float64 QuSpin ED of the quantax operator's static list, compared
     against the numba ED reference JSON (independent code path).
  3. Stage 1: unprojected ResConv + SR.  Stage 2: symmetry-projected fine-tuning.
  4. Exact dense evaluation (float64): NQS energy, relative error, fidelity.
  5. Print a single ``RESULT_JSON:`` line and write results.json.

Invoked by scripts/colab_remote_runner.py when the config declares
``runner: mixd.train_quantax``; also runnable directly:
    PYTHONPATH=src:research/src python -u -m mixd.train_quantax --config research/configs/<cfg>.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from mixd.lattice import HeisenbergCouplings, MultilayerLattice
from mixd.quantax_ops import mixd_heisenberg_operator

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = REPO_ROOT / "research"


def _to_numpy_psi(x) -> np.ndarray:
    if hasattr(x, "logabs"):
        logabs = np.asarray(x.logabs, dtype=np.float64)
        sign = np.asarray(x.sign)
        return np.real(sign * np.exp(logabs - logabs.max())).astype(np.float64)
    return np.real(np.asarray(x)).astype(np.float64)


def _build_symmetry(names: list[str]):
    from quantax.symmetry import Identity, SpinInverse

    table = {"spin_inverse": SpinInverse}
    symm = Identity()
    for n in names:
        if n not in table:
            raise ValueError(f"Unsupported projection symmetry '{n}'. Options: {sorted(table)}")
        symm = symm @ table[n]()
    return symm


def run(cfg: dict, ckpt_root: Path) -> dict:
    import jax
    import jax.numpy as jnp
    import quantax as qtx
    from quantax.symmetry import Identity
    from quspin.operators import hamiltonian as quspin_hamiltonian

    sysc, modc, trc = cfg["system"], cfg.get("model", {}), cfg.get("training", {})
    name = cfg.get("telemetry", {}).get("experiment_name", "mixd_run")
    out_dir = ckpt_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    lat = MultilayerLattice(
        lx=sysc["lx"], ly=sysc["ly"], n_layers=sysc["n_layers"],
        pbc_x=bool(sysc.get("pbc", [True, True])[0]), pbc_y=bool(sysc.get("pbc", [True, True])[1]),
    )
    if not lat.is_bipartite():
        raise ValueError("Marshall-sign runner requires a bipartite cluster")
    cpl = HeisenbergCouplings.build(lat, sysc["j_par"], sysc["j_perp"])
    n = lat.n_sites
    qtx.set_random_seed(int(trc.get("seed", 42)))

    print("=" * 72, flush=True)
    print(f"[mixd] experiment : {name}", flush=True)
    print(f"[mixd] devices    : {jax.devices()}", flush=True)
    print(f"[mixd] cluster    : {lat.label()} (N={n}, bonds={len(cpl.bond_list)}), "
          f"J_par={cpl.j_par}, J_perp={cpl.j_perp}, msr={sysc.get('msr', True)}", flush=True)
    print("=" * 72, flush=True)

    boundary = [1 if lat.pbc_x else 0, 1 if lat.pbc_y else 0, 0]
    qtx.sites.Grid([lat.lx, lat.ly, lat.n_layers], boundary=boundary, Nparticles=(n // 2, n // 2))
    op = mixd_heisenberg_operator(cpl.bond_list, msr=bool(sysc.get("msr", True)))

    # ---- operator check (float64, no symmetry, Sz=0 sector) -------------------------
    compute_exact = bool(sysc.get("compute_exact", n <= 24))
    ident = Identity()
    ref = None
    ref_path = sysc.get("reference")
    if ref_path and ref_path != "none":
        p = RESEARCH_ROOT / ref_path
        if p.exists():
            ref = json.loads(p.read_text(encoding="utf-8"))
    check_ok = None
    e_exact = ref["E0"] if ref is not None else None
    h64 = psi0 = basis = None
    if compute_exact:
        ident.basis_make()
        basis = ident.basis
        no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)
        h64 = quspin_hamiltonian(op.quspin_static_list, [], basis=basis, dtype=np.float64, **no_checks)
        t0 = time.time()
        w, v = h64.eigsh(k=1, which="SA")
        e_qtx = float(w[0])
        psi0 = v[:, 0] / np.linalg.norm(v[:, 0])
        print(f"[check] quantax-operator ED (float64, dim={basis.Ns}): E0 = {e_qtx:.10f} ({time.time() - t0:.1f}s)", flush=True)
        if ref is not None:
            diff = e_qtx - ref["E0"]
            check_ok = abs(diff) < 1e-8
            print(f"[check] numba ED reference E0 = {ref['E0']:.10f} | diff = {diff:.2e} -> "
                  f"{'PASS' if check_ok else 'FAIL'}", flush=True)
            if not check_ok:
                raise RuntimeError("quantax operator does not reproduce the ED reference")
        e_exact = e_qtx
    else:
        print("[check] cluster too large for ED; skipping operator check and dense evaluation", flush=True)

    # ---- model ----------------------------------------------------------------------
    kwargs = dict(nblocks=modc.get("nblocks", 2), channels=modc.get("channels", 8),
                  kernel_size=modc.get("kernel_size", 3))
    if modc.get("final_activation", "exp") == "sinhp1_by_log":
        kwargs["final_activation"] = qtx.nn.sinhp1_by_log
    model = qtx.model.ResConv(**kwargs)

    nsamples = int(trc.get("nsamples", 1024))
    neighbors = trc.get("sampler_neighbors", [1])
    log_every = int(cfg.get("telemetry", {}).get("log_every", 25))
    curve: list[list[float]] = []

    def train(state, sampler, steps, lr, stage):
        opt = qtx.optimizer.SR(state, op)
        tic = time.time()
        for step in range(1, steps + 1):
            samples = sampler.sweep()
            upd = opt.get_step(samples)
            state.update(upd * lr)
            e = float(np.asarray(opt.energy).real)
            var = float(np.asarray(opt.VarE).real)
            if math.isnan(e) or math.isnan(var):
                raise RuntimeError(f"NaN at stage {stage} step {step}")
            curve.append([stage, step, e, var])
            if step % log_every == 0 or step == steps:
                dt = (time.time() - tic) / log_every
                tic = time.time()
                rel = f" | rel_err = {abs(e - e_exact) / abs(e_exact):.2e}" if e_exact is not None else ""
                print(f"[Stage{stage} {step:4d}/{steps}] E = {e:.6f} | VarE = {var:.3e} | "
                      f"V-score = {n * var / e**2:.3e}{rel} ({dt * 1e3:.0f} ms/step)",
                      flush=True)

    t_train = time.time()
    state1 = qtx.state.Variational(model, max_parallel=int(trc.get("max_parallel_stage1", 20000)))
    print(f"[model] parameters: {state1.nparams:,}", flush=True)
    sampler1 = qtx.sampler.SpinExchange(state1, nsamples=nsamples,
                                        reweight=float(trc.get("reweight", 2.0)), n_neighbor=neighbors)
    train(state1, sampler1, int(trc.get("stage1_steps", 300)), float(trc.get("learning_rate", 5e-3)), 1)
    ckpt1 = out_dir / "stage1.eqx"
    state1.save(ckpt1)
    chains = qtx.utils.to_replicated_numpy(sampler1._spins)

    proj = _build_symmetry(modc.get("projection", ["spin_inverse"]))
    state2 = qtx.state.Variational(model, param_file=ckpt1, symm=proj,
                                   max_parallel=int(trc.get("max_parallel_stage2", 4096)))
    sampler2 = qtx.sampler.SpinExchange(state2, nsamples=nsamples,
                                        reweight=float(trc.get("reweight", 2.0)), n_neighbor=neighbors,
                                        initial_spins=jnp.asarray(chains))
    train(state2, sampler2, int(trc.get("stage2_steps", 300)),
          float(trc.get("stage2_learning_rate", trc.get("learning_rate", 5e-3))), 2)
    state2.save(out_dir / "final.eqx")
    train_time = time.time() - t_train

    # ---- exact dense evaluation (float64) -------------------------------------------
    e_dense = fidelity = rel_dense = None
    if compute_exact:
        psi = _to_numpy_psi(state2[basis.states])
        psi /= np.linalg.norm(psi)
        e_dense = float(psi @ h64.dot(psi))
        fidelity = float((psi @ psi0) ** 2)
        rel_dense = abs(e_dense - e_exact) / abs(e_exact)
        print(f"[dense] E_NQS = {e_dense:.10f} | E_exact = {e_exact:.10f} | rel_err = {rel_dense:.3e} | "
              f"fidelity = {fidelity:.8f}", flush=True)

    result = {
        "experiment": name, "cluster": lat.label(), "n_sites": n,
        "j_par": cpl.j_par, "j_perp": cpl.j_perp, "convention": "spin-1/2 S.S",
        "operator_check_pass": check_ok, "E_exact": e_exact,
        "E_nqs_dense": e_dense, "rel_err_dense": rel_dense, "fidelity": fidelity,
        "E_last_stage1": curve[int(trc.get("stage1_steps", 300)) - 1][2], "E_last_stage2": curve[-1][2],
        "nparams": int(state1.nparams), "train_time_s": round(train_time, 1),
        "devices": str(jax.devices()),
    }
    np.savetxt(out_dir / "training_curve.txt", np.asarray(curve), header="stage step energy VarE")
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_JSON: " + json.dumps(result), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default=str(RESEARCH_ROOT / "runs"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args, _ = ap.parse_known_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run(cfg, Path(args.checkpoint_dir))


if __name__ == "__main__":
    main()
