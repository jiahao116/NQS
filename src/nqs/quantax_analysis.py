"""Scientific verification utilities for Quantax NQS runs.

Implements production-grade post-training diagnostics distilled from the
workspace skills (`measure`, `symm-test`, `zero-var-extrap`):
  - Streaming autocorrelation binning analysis (`BinningAccumulator`)
  - Symmetry overlap diagnostic O_T (`compute_symmetry_overlaps`)
  - Orthogonal distance regression zero-variance extrapolation (`run_zero_variance_extrapolation`)
  - Multi-panel publication figure renderer (`save_quantax_results_panel`)
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any
import warnings

if TYPE_CHECKING:
    import numpy as np

R_MAX = 10.0  # Lever-arm threshold above which the ODR slope is weakly constrained


class BinningAccumulator:
    """Streaming O(log N_sweeps) binning analysis of correlated VMC local estimators.

    Feed one batch of shape `(nchains,)` or `(nchains, nobs)` per sweep via `add`.
    Estimators are centered by the first-sweep estimate O0 before weighting by `r`
    so float32 accumulation avoids catastrophic cancellation and `<O>^2 Var(r)` bias.
    """

    def __init__(self) -> None:
        self._levels: list[dict[str, Any]] = []
        self._offset: Any = None

    def add(self, oloc: Any, r: Any = None) -> None:
        import jax.numpy as jnp

        x = jnp.asarray(oloc)
        if x.ndim == 1:
            x = x[:, None]
        if r is not None:
            r = jnp.asarray(r)[:, None]
        if self._offset is None:
            self._offset = (x if r is None else r * x).mean(axis=0)
        x = x - self._offset
        if r is not None:
            x = r * x
        lvl = 0
        while x is not None:
            if lvl == len(self._levels):
                self._levels.append({"buf": None, "s1": 0.0, "s2": 0.0, "n": 0})
            level = self._levels[lvl]
            level["s1"] = level["s1"] + x
            level["s2"] = level["s2"] + x**2
            level["n"] += 1
            if level["buf"] is None:
                level["buf"] = x
                x = None
            else:
                x = (level["buf"] + x) / 2.0
                level["buf"] = None
            lvl += 1

    def results(self, min_bins: int = 8) -> dict[str, Any]:
        import jax.numpy as jnp
        import numpy as np

        bin_sizes: list[int] = []
        errors: list[np.ndarray] = []
        for lvl, level in enumerate(self._levels):
            if level["n"] < min_bins:
                break
            m = level["n"] * level["s1"].shape[0]
            mean_l = level["s1"].sum(axis=0) / m
            var_l = jnp.maximum(level["s2"].sum(axis=0) / m - mean_l**2, 0.0)
            errors.append(np.asarray(jnp.sqrt(var_l / max(m - 1, 1))))
            bin_sizes.append(2**lvl)
        if not errors:
            raise ValueError(f"Binning requires nsweeps >= min_bins ({min_bins}).")

        err_arr = np.stack(errors)
        l0 = self._levels[0]
        mean = np.asarray(self._offset + l0["s1"].sum(axis=0) / (l0["n"] * l0["s1"].shape[0]))
        e0 = np.maximum(err_arr[0], 1e-30)
        tau = 0.5 * ((err_arr[-1] / e0) ** 2 - 1.0)
        plateaued = True
        if len(err_arr) >= 2 and np.any(err_arr[-1] > 1.2 * err_arr[-2]):
            plateaued = False
        return {
            "mean": float(mean[0]),
            "err": float(err_arr[-1][0]),
            "tau_int": float(tau[0]),
            "bin_sizes": [int(b) for b in bin_sizes],
            "bin_errors": [float(e[0]) for e in err_arr],
            "plateaued": bool(plateaued),
        }


def build_staggered_mag_sq_operator(lattice: Any) -> Any:
    """Construct diagonal staggered magnetization squared operator M_stag^2 in-place.

    M_stag = (1 / N) * sum_{x, y} (-1)^(x + y) sigma^z_{x, y},
    so M_stag^2 equals the Neel antiferromagnetic structure factor S_z(pi, pi) / N.
    Uses in-place `+=` to avoid quadratic Operator.__add__ overhead.
    """
    from quantax.operator import Operator, sigma_z

    m_stag = Operator([])
    n_sites = lattice.Nsites
    if len(lattice.shape) >= 3:
        lx, ly = int(lattice.shape[1]), int(lattice.shape[2])
        for x in range(lx):
            for y in range(ly):
                sign = 1.0 if (x + y) % 2 == 0 else -1.0
                m_stag += sign * sigma_z(x, y)
    else:
        for i in range(n_sites):
            sign = 1.0 if i % 2 == 0 else -1.0
            m_stag += sign * sigma_z(i)
    m_stag /= float(n_sites)
    return m_stag @ m_stag


def compute_symmetry_overlaps(state: Any, samples: Any, symm: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute squared normalized symmetry overlap O_T and error per group element.

    O_T = <psi|T|psi><psi|T^-1|psi> / <psi|psi>^2.
    Handles spin Z2 inversion (`symm.Z2_inversion != 0`), fermion permutation signs,
    reweighting weights `samples.r`, and correlated covariance error propagation.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from quantax.symmetry.symmetry import _permutation_sign

    spins = samples.spins
    nsamples = spins.shape[0]
    perm = jnp.asarray(symm._perm)
    ngroup = perm.shape[0]

    perm_np = np.asarray(symm._perm, dtype=np.int64)
    lookup = {row.tobytes(): i for i, row in enumerate(perm_np)}
    inv_idx = [lookup[row.tobytes()] for row in np.argsort(perm_np, axis=1)]

    if symm.is_fermion:
        if symm.Z2_inversion != 0:
            raise NotImplementedError("Particle-hole fermion sign not implemented.")
        sign_fn = jax.vmap(_permutation_sign, in_axes=(0, None, None))
        sign_symm = sign_fn(spins, perm, symm._perm_sign)
    else:
        sign_symm = jnp.ones((nsamples, ngroup))

    r = 1.0 if samples.r is None else samples.r
    z2_flips = (1,) if symm.Z2_inversion == 0 else (1, -1)

    overlaps: list[float] = []
    errors: list[float] = []
    for z2 in z2_flips:
        xs = []
        for g in range(ngroup):
            # Form ratio on LogArrays before converting to avoid fp32 overflow/underflow
            ratio = state(z2 * spins[:, perm[g]]) / samples.psi
            xs.append(sign_symm[:, g] * jnp.asarray(ratio))
        for g in range(ngroup):
            x, y = xs[g], xs[inv_idx[g]]
            xmean = jnp.mean(r * x)
            ymean = jnp.mean(r * y)
            dxy = jnp.stack([r * (x - xmean), r * (y - ymean)])
            cov = jnp.cov(dxy) / nsamples
            var = ymean**2 * cov[0, 0] + xmean**2 * cov[1, 1] + 2 * xmean * ymean * cov[0, 1]
            overlaps.append(float((xmean * ymean).real))
            errors.append(float(jnp.sqrt(jnp.abs(var)).real))

    return np.asarray(overlaps, dtype=float), np.asarray(errors, dtype=float)


def run_symmetry_diagnostics(
    model: Any,
    param_file: Path,
    symm_projected: Any | None,
    initial_spins: np.ndarray | None,
    nsamples: int = 1024,
    n_neighbor: int | list[int] = 1,
) -> dict[str, list[dict[str, float]]]:
    """Measure symmetry overlaps O_T on key square-lattice symmetry generators."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import quantax as qtx
    from quantax.symmetry import Flip, Rotation, SpinInverse, Translation

    n_dev = max(1, jax.device_count())
    n_eff = max(n_dev, (nsamples // n_dev) * n_dev)
    max_parallel = n_eff // n_dev

    eval_state = qtx.state.Variational(
        model,
        param_file=param_file,
        symm=symm_projected,
        max_parallel=max_parallel,
    )

    init_arr = _resize_chains(initial_spins, n_eff)
    sampler_kwargs: dict[str, Any] = {
        "nsamples": n_eff,
        "n_neighbor": n_neighbor,
    }
    if init_arr is not None:
        sampler_kwargs["initial_spins"] = jnp.asarray(init_arr)
        sampler_kwargs["thermal_steps"] = 5 * qtx.get_sites().Nmodes

    sampler = qtx.sampler.SpinExchange(eval_state, **sampler_kwargs)
    s = sampler.sweep()
    samples = qtx.sampler.Samples(s.spins, psi=eval_state(s.spins), prob=s.prob)
    samples.update_r()

    candidates = {
        "trans_x": Translation((1, 0)),
        "rot90": Rotation(np.pi / 2),
        "flip_x": Flip(axis=0),
        "spin_inv": SpinInverse(),
        "flip_x_spin_inv": Flip(axis=0) @ SpinInverse(),
    }

    report: dict[str, list[dict[str, float]]] = {}
    for name, symm_op in candidates.items():
        ov, err = compute_symmetry_overlaps(eval_state, samples, symm_op)
        report[name] = [
            {"element": int(idx), "overlap": float(o), "error": float(e)}
            for idx, (o, e) in enumerate(zip(ov, err))
        ]
    return report


def _resize_chains(chains: Any | None, target_rows: int) -> Any | None:
    if chains is None:
        return None
    import numpy as np

    arr = np.asarray(chains)
    if arr.shape[0] == target_rows:
        return arr
    if arr.shape[0] > target_rows:
        idx = np.linspace(0, arr.shape[0] - 1, target_rows, dtype=int)
        return arr[idx]
    reps = int(np.ceil(target_rows / arr.shape[0]))
    return np.tile(arr, (reps, 1))[:target_rows]


def run_binning_measurement(
    state: Any,
    hamiltonian: Any,
    m_stag2_op: Any,
    initial_spins: np.ndarray | None,
    nchains: int = 512,
    nsweeps: int = 64,
    sweep_steps: int | None = None,
    n_neighbor: int | list[int] = 1,
) -> dict[str, Any]:
    """Run thermalized production measurement with streaming autocorrelation binning."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import quantax as qtx

    n_modes = qtx.get_sites().Nmodes
    n_dev = max(1, jax.device_count())
    nchains_eff = max(n_dev, (nchains // n_dev) * n_dev)
    sw_steps = sweep_steps if sweep_steps is not None else 2 * n_modes

    init_arr = _resize_chains(initial_spins, nchains_eff)
    sampler_kwargs: dict[str, Any] = {
        "nsamples": nchains_eff,
        "n_neighbor": n_neighbor,
        "sweep_steps": sw_steps,
    }
    if init_arr is not None:
        sampler_kwargs["initial_spins"] = jnp.asarray(init_arr)
        sampler_kwargs["thermal_steps"] = 5 * n_modes
    else:
        sampler_kwargs["thermal_steps"] = 40 * n_modes

    sampler = qtx.sampler.SpinExchange(state, **sampler_kwargs)

    acc_e = BinningAccumulator()
    acc_m2 = BinningAccumulator()
    e_sweeps: list[Any] = []
    vare_sweeps: list[Any] = []
    n_zero_r = 0

    for _ in range(nsweeps):
        samples = sampler.sweep()
        # Off-diagonal first so H.Oloc populates samples.psi and samples.r in place
        e_loc = hamiltonian.Oloc(state, samples).real
        r = samples.r
        w = jnp.ones_like(e_loc) if r is None else r
        acc_e.add(e_loc, r)
        e_mean_sw = jnp.mean(w * e_loc)
        e_sweeps.append(e_mean_sw)
        vare_sweeps.append(jnp.mean(w * (e_loc - e_mean_sw) ** 2))
        n_zero_r = n_zero_r + int(jnp.sum(w == 0))

        # Diagonal staggered magnetization squared reuses spins & r at negligible cost
        m2_loc = m_stag2_op.apply_diag(samples.spins).real
        acc_m2.add(m2_loc, r)

    e_res = acc_e.results(min_bins=8)
    m2_res = acc_m2.results(min_bins=8)
    e_sw_np = np.asarray(jnp.stack(e_sweeps), dtype=float)
    v_sw_np = np.asarray(jnp.stack(vare_sweeps), dtype=float)

    scatter = float(e_sw_np.std())
    expected_scatter = float(np.sqrt(max(v_sw_np.mean(), 0.0) / nchains_eff))

    return {
        "nchains": int(nchains_eff),
        "nsweeps": int(nsweeps),
        "total_samples": int(nchains_eff * nsweeps),
        "energy": e_res,
        "m_stag2": m2_res,
        "var_e_mean": float(v_sw_np.mean()),
        "sweep_scatter": scatter,
        "expected_scatter": expected_scatter,
        "n_zero_r": int(n_zero_r),
    }


def _bin_series(series: Any, bin_size: int) -> tuple[Any, Any]:
    import numpy as np

    arr = np.asarray(series, dtype=float)
    n = (arr.size // bin_size) * bin_size
    binned = arr[:n].reshape(-1, bin_size)
    return binned.mean(axis=1), binned.std(axis=1) / np.sqrt(bin_size)


def _lever_arm(v: Any, nchunks: int = 4) -> float:
    import numpy as np

    arr = np.asarray(v, dtype=float)
    if len(arr) < nchunks:
        return float("nan")
    medians = [float(np.median(c)) for c in np.array_split(arr, nchunks)]
    ptp = float(np.ptp(medians))
    if ptp <= 0.0:
        return float("inf")
    return float(np.median(arr)) / ptp


def run_zero_variance_extrapolation(
    curve_data: Any,
    starts: list[int] | None = None,
    ends: list[int] | None = None,
    bins: list[int] | None = None,
    maxit: int = 300,
) -> dict[str, Any]:
    """Run ODR zero-variance extrapolation (`odrpack`) with window-grid error bars.

    `curve_data` has shape `(N, 3)` with columns `[step, E, VarE]`.
    """
    import numpy as np
    from odrpack import odr_fit

    data = np.asarray(curve_data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3 or len(data) < 40:
        return {"available": False, "reason": "Insufficient curve length for ODR extrapolation."}

    steps = data[:, 0]
    min_s, max_s = int(steps[0]), int(steps[-1]) + 1
    span = max_s - min_s

    if starts is None:
        starts = sorted({min_s + int(span * f) for f in (0.10, 0.20, 0.30, 0.40)})
    if ends is None:
        ends = sorted({min_s + int(span * f) for f in (0.85, 0.95, 1.00)})
    if bins is None:
        bins = [b for b in (8, 12, 16, 20) if span // b >= 5]
        if not bins:
            bins = [max(4, span // 8)]

    rows: list[dict[str, Any]] = []
    best_plot_pack: dict[str, Any] | None = None
    best_score = float("inf")

    for start, end, bin_size in itertools.product(starts, ends, bins):
        mask = (steps >= start) & (steps < end)
        sub = data[mask]
        if sub.shape[0] // bin_size < 4:
            continue
        y, y_err = _bin_series(sub[:, 1], bin_size)
        x, x_err = _bin_series(sub[:, 2], bin_size)
        s_bin, _ = _bin_series(sub[:, 0], bin_size)
        if np.any(x_err <= 0) or np.any(y_err <= 0) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            continue
        if np.std(x) <= 1e-15:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta0 = np.polyfit(x, y, 1)
            res = odr_fit(
                lambda xv, beta: beta[0] * xv + beta[1],
                x,
                y,
                beta0=beta0,
                weight_x=1.0 / (x_err**2),
                weight_y=1.0 / (y_err**2),
                maxit=maxit,
            )

        k, b = float(res.beta[0]), float(res.beta[1])
        k_err, b_err = float(res.sd_beta[0]), float(res.sd_beta[1])
        res_var = float(res.res_var)
        r_arm = _lever_arm(sub[:, 2])
        extrap_dist = abs(k) * float(np.mean(x))
        ok = bool(res.success and k > 0.0 and b <= float(np.min(y)))

        rows.append(
            {
                "start": int(start),
                "end": int(end),
                "bin": int(bin_size),
                "slope": k,
                "intercept": b,
                "sd_intercept": b_err,
                "res_var": res_var,
                "lever_arm": r_arm,
                "extrap_dist": extrap_dist,
                "kept": ok,
            }
        )

        # Prefer a clean, well-identified fit with finite lever arm for visualization
        if ok and np.isfinite(b_err) and np.isfinite(r_arm):
            score = b_err * (1.0 + max(0.0, r_arm - 5.0) * 0.1)
            if score < best_score:
                best_score = score
                best_plot_pack = {
                    "start": int(start),
                    "end": int(end),
                    "bin": int(bin_size),
                    "k": k,
                    "b": b,
                    "k_err": k_err,
                    "b_err": b_err,
                    "res_var": res_var,
                    "lever_arm": r_arm,
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "x_err": x_err.tolist(),
                    "y_err": y_err.tolist(),
                    "step": s_bin.tolist(),
                }

    if not rows:
        return {"available": False, "reason": "No valid ODR windows produced."}

    kept_rows = [r for r in rows if r["kept"]]
    if kept_rows:
        med_sd = float(np.median([r["sd_intercept"] for r in kept_rows]))
        for r in rows:
            if r["kept"] and r["sd_intercept"] > 5.0 * med_sd:
                r["kept"] = False
        kept_rows = [r for r in rows if r["kept"]]

    if len(kept_rows) < 2:
        return {
            "available": False,
            "reason": f"Only {len(kept_rows)}/{len(rows)} windows passed variational/convergence checks.",
            "n_total_windows": len(rows),
            "n_kept_windows": len(kept_rows),
            "best_window": best_plot_pack,
        }

    intercepts = np.array([r["intercept"] for r in kept_rows], dtype=float)
    sds = np.array([r["sd_intercept"] for r in kept_rows], dtype=float)
    r_arms = np.array([r["lever_arm"] for r in kept_rows], dtype=float)
    weak_count = int(sum(1 for r in kept_rows if r["sd_intercept"] > 0.5 * r["extrap_dist"]))

    return {
        "available": True,
        "e0_mean": float(intercepts.mean()),
        "stat_err": float(sds.mean()),
        "syst_err": float(intercepts.std(ddof=1)),
        "median_lever_arm": float(np.nanmedian(r_arms)),
        "median_res_var": float(np.median([r["res_var"] for r in kept_rows])),
        "n_kept_windows": len(kept_rows),
        "n_total_windows": len(rows),
        "n_weak_windows": weak_count,
        "weakly_constrained": bool(np.nanmedian(r_arms) > R_MAX or weak_count > len(kept_rows) // 2),
        "best_window": best_plot_pack,
    }


def save_quantax_results_panel(
    stage1_curve: np.ndarray,
    stage2_curve: np.ndarray,
    exact_energy: float | None,
    dense_energy: float | None,
    dense_fidelity: float | None,
    zve_result: dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    """Render a publication-quality 3-panel NQS benchmark figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    s1 = np.asarray(stage1_curve, dtype=float)
    s2 = np.asarray(stage2_curve, dtype=float)
    s1_len = len(s1)

    steps_all = np.concatenate([s1[:, 0], s2[:, 0] + s1_len])
    e_all = np.concatenate([s1[:, 1], s2[:, 1]])
    v_all = np.concatenate([s1[:, 2], s2[:, 2]])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=180)
    fig.patch.set_facecolor("white")

    # --- Panel A: Energy convergence across Stage 1 -> Stage 2 ---
    ax0 = axes[0]
    ax0.plot(s1[:, 0], s1[:, 1], color="#2563eb", alpha=0.45, linewidth=1.0, label="Stage 1 (ResConv)")
    ax0.plot(s2[:, 0] + s1_len, s2[:, 1], color="#059669", alpha=0.55, linewidth=1.0, label="Stage 2 (+ $C_{4v}\\times\\mathbb{Z}_2$ Symm)")

    # Smooth running mean (window 10) for visual clarity
    win = min(10, max(1, len(e_all) // 20))
    if win > 1:
        kernel = np.ones(win) / win
        e_smooth = np.convolve(e_all, kernel, mode="valid")
        ax0.plot(steps_all[win - 1 :], e_smooth, color="#111827", linewidth=1.6, label=f"Moving Avg (w={win})")

    ax0.axvline(s1_len, color="#6b7280", linestyle=":", linewidth=1.3, label="Symmetry Projection On")
    if exact_energy is not None:
        ax0.axhline(exact_energy, color="#dc2626", linestyle="--", linewidth=1.6, label=f"Exact ED = {exact_energy:.5f}")

    ax0.set_xlabel("VMC Optimization Step")
    ax0.set_ylabel("Energy $E$")
    ax0.set_title("(a) Ground-State Energy Convergence")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", frameon=True, framealpha=0.9)

    # --- Panel B: Energy Variance & Relative Error (log scale) ---
    ax1 = axes[1]
    v_pos = np.maximum(v_all, 1e-12)
    ax1.semilogy(steps_all, v_pos, color="#7c3aed", alpha=0.75, linewidth=1.2, label="$\\mathrm{Var}(E)$")
    if exact_energy is not None:
        rel_err = np.maximum(np.abs(e_all - exact_energy) / max(abs(exact_energy), 1e-12), 1e-10)
        ax1.semilogy(steps_all, rel_err, color="#ea580c", alpha=0.80, linewidth=1.2, label="$|E - E_0| / |E_0|$")
    ax1.axvline(s1_len, color="#6b7280", linestyle=":", linewidth=1.3)

    if dense_energy is not None and exact_energy is not None:
        final_rel = abs(dense_energy - exact_energy) / max(abs(exact_energy), 1e-12)
        badge = f"Final Exact Rel. Err: {final_rel:.2e}"
        if dense_fidelity is not None:
            badge += f"\nWavefunction Fidelity: {dense_fidelity:.6f}"
        ax1.text(
            0.04,
            0.06,
            badge,
            transform=ax1.transAxes,
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )

    ax1.set_xlabel("VMC Optimization Step")
    ax1.set_ylabel("Log Scale")
    ax1.set_title("(b) Energy Variance & Relative Error")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    # --- Panel C: Zero-Variance ODR Extrapolation ---
    ax2 = axes[2]
    bw = zve_result.get("best_window")
    if bw is not None:
        x = np.asarray(bw["x"], dtype=float)
        y = np.asarray(bw["y"], dtype=float)
        x_err = np.asarray(bw["x_err"], dtype=float)
        y_err = np.asarray(bw["y_err"], dtype=float)
        st = np.asarray(bw["step"], dtype=float)
        k, b, b_err = float(bw["k"]), float(bw["b"]), float(bw["b_err"])

        ax2.errorbar(
            x,
            y,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            ecolor="#9ca3af",
            alpha=0.7,
            capsize=2,
            zorder=1,
        )
        sc = ax2.scatter(x, y, s=22, c=st, cmap="viridis", zorder=3)
        cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
        cbar.set_label("Stage 2 Step", fontsize=8.5)

        x_max = max(1.15 * float(np.quantile(x, 0.97)), 1e-6)
        x_line = np.linspace(0.0, x_max, 120)
        ax2.plot(
            x_line,
            k * x_line + b,
            "k--",
            linewidth=1.5,
            label=f"ODR $E_0$ = {b:.5f} $\\pm$ {b_err:.5f}",
            zorder=2,
        )
        if exact_energy is not None:
            ax2.scatter(
                [0.0],
                [exact_energy],
                marker="*",
                s=130,
                color="#dc2626",
                zorder=5,
                label=f"Exact ED = {exact_energy:.5f}",
            )
        ax2.set_xlim(-0.04 * x_max, x_max)
        r_arm = bw.get("lever_arm", float("nan"))
        ax2.set_title(f"(c) Zero-Variance ODR Extrapolation ($R={r_arm:.1f}$)")
        ax2.legend(loc="lower right", frameon=True, framealpha=0.9)
    else:
        ax2.text(
            0.5,
            0.5,
            "ODR window not identified\n(already converged near VarE -> 0)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=9.5,
            color="#475569",
        )
        ax2.set_title("(c) Zero-Variance Extrapolation")

    ax2.set_xlabel("Energy Variance $\\mathrm{Var}(E)$")
    ax2.set_ylabel("Energy $E$")
    ax2.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
