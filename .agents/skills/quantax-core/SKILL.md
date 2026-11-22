---
name: quantax-core
description: Core workflow, architecture, sharp-bit safeguards, and Colab/W&B integration rules for running Neural Quantum States (NQS) simulations with Quantax (v0.4.0+, JAX + Equinox). Use whenever designing, implementing, debugging, or benchmarking a Quantax VMC training or measurement script (e.g., 2D Heisenberg, J1-J2, triangular, or Hubbard models).
---

# Quantax Core Workflow & Safeguards

Quantax (`ChenAo-Phys/quantax`, v0.4.0+) is a research-oriented JAX/Equinox framework for scalable 2D/3D lattice NQS with native MinSR/SPRING optimization, space-group symmetry projection, exact diagonalization via QuSpin, and fermionic backflow/Pfaffian wavefunctions.

## 1. Layered VMC Pipeline

Every Quantax simulation composes objects in this strict order:

```text
sites (global) -> operator -> symmetry -> model -> state -> sampler -> optimizer -> loop
```

### Key Building Blocks
- **Lattices (`quantax.sites`)**: `Chain`, `Square`, `Grid`, `Cube`, `Triangular`, `TriangularB`, `Pyrochlore`.
  - Conserve spin magnetization with `Nparticles=(N_up, N_dn)` (e.g. `qtx.sites.Square(4, Nparticles=(8, 8))`).
  - For fermions: `particle_type=qtx.PARTICLE_TYPE.spinful_fermion`.
- **Hamiltonians (`quantax.operator`)**: `Ising`, `Heisenberg`, `Hubbard`, `tJ`, `tV`.
  - Frustrated Square $J_1$-$J_2$: `qtx.operator.Heisenberg(J=[1.0, 0.5], n_neighbor=[1, 2], msr=True)`.
  - Exact Diagonalization benchmark (small clusters, $N \le 20$): `E, wf = H.diagonalize(full_symm)`.
- **Symmetries (`quantax.symmetry`)**: `TransND`, `Translation`, `Rotation`, `Flip`, `C4v`, `D6`, `SpinInverse`, `Z2Inversion`, `ParticleHole`.
  - Compose groups with `@`: `TransND() @ Rotation(np.pi / 2) @ Flip() @ SpinInverse()`.
  - Because `ResConv` is already translation-invariant, project only the remaining point-group & spin-flip symmetries: `symm = Rotation(np.pi / 2) @ Flip() @ SpinInverse()`.
- **Models (`quantax.model`)**:
  - `ResConv(nblocks, channels, kernel_size, final_activation=...)` — workhorse 2D deep residual CNN.
  - `RBM_Conv(channels)`, `RBM_Dense(features)` — shallow RBMs.
  - `DetBackflow`, `PfBackflow`, `GeneralDet`, `GeneralPf`, `GeneralJastrow` — fermionic & correlated ansatzes.
- **Samplers (`quantax.sampler`)**:
  - `SpinExchange(state, nsamples, n_neighbor=..., reweight=...)` for $S_z$-conserving spin models.
  - `LocalFlip` for Ising/unconstrained spins; `ParticleHop` for Hubbard/fermions; `MixSampler` to combine local + global moves.
- **Optimizers (`quantax.optimizer`)**:
  - `SR(state, H, solver=...)` — uses `auto_shift_eig` by default, which automatically switches to the **MinSR** dual formulation when `nsamples < nparams` (scaling as $O(N_s^3)$ instead of $O(N_p^3)$).
  - Alternatives: `SPRING`, `MARCH`, `AdamSR`, `ER` (exact reconfiguration on tiny clusters), `TimeEvol`.
  - For very large `nsamples` or TPU/memory-tight runs, prefer `solver=qtx.optimizer.lsmr()`.

## 2. Sign Structure Recipes

Choosing the right sign/phase structure is essential for 2D frustrated magnets:

1. **Bipartite unfrustrated Heisenberg ($J_2 = 0$)**:
   - Exact Marshall Sign Rule (MSR): set `Heisenberg(msr=True)` so the ground state is strictly positive in the rotated basis.
   - Use default positive `ResConv` (`exp` final activation).
2. **Frustrated real-valued ground state (e.g., Square $J_1$-$J_2$ at $J_2 / J_1 = 0.5$)**:
   - Keep `msr=True` as a strong approximate sign prior, AND allow negative amplitudes via `final_activation=qtx.nn.sinhp1_by_log` (overflow-safe implementation of $\sinh(x) + 1$).
3. **Complex / non-collinear sign structure (e.g., Triangular antiferromagnet)**:
   - Call `qtx.set_default_dtype(jnp.complex64)` before constructing the model, and initialize with a classical reference phase (e.g., $120^\circ$ Néel pattern) when helpful.

## 3. Two-Stage Symmetry Projection Protocol

Do **not** turn on expensive full-group projection from step 0:

1. **Stage 1 (Unprojected pre-training)**: Train translation-invariant `ResConv` with a large `max_parallel` (e.g. `20000` on 4x4) until energy and `VarE` stabilize. Save weights (`state.save("stage1.eqx")`) and thermalized chains (`np.save("chains_stage1.npy", qtx.utils.to_replicated_numpy(sampler._spins))`).
2. **Symmetry diagnostic**: Run the [symm-test](../symm-test/SKILL.md) skill (`O_T` overlap test) to confirm the candidate symmetries are approximately satisfied ($O_T \sim 0.9$, not broken with $O_T \approx 0$).
3. **Stage 2 (Symmetry-projected fine-tuning)**: Load `"stage1.eqx"` into `qtx.state.Variational(model, "stage1.eqx", symm=symm, max_parallel=...)`, scaling `max_parallel` down by `~symm.nsymm` to stay within GPU VRAM! Pass saved chains via `initial_spins=` (or re-thermalize generously) so `VarE` does not spike at restart.

## 4. Non-Negotiable Sharp Bits (Avoid Silent Bugs!)

1. **One `Sites` per Python process**: `Sites._SITES` is a process-global singleton. Never construct two different lattices in the same Python process (switching systems requires a fresh subprocess — which matches `scripts/colab_remote_runner.py`!).
2. **`max_parallel` chunking & padding**:
   - Always pass an explicit `max_parallel` to `Variational` (or a `(forward_chunk, backward_chunk)` tuple).
   - Under `symm=`, every input configuration expands to `symm.nsymm` images internally — **never multiply `max_parallel` by `symm.nsymm`**; instead divide it by `symm.nsymm`.
   - Choose `backward_chunk` to divide `nsamples // ndevices` to avoid padded-copy memory spikes during Jacobian evaluation.
3. **Precision (TF32 vs FP32 vs FP64)**:
   - On NVIDIA GPUs (L4 / A100), JAX defaults to TF32 matmuls. **RBM is numerically unstable under TF32** — always disable TF32 via `jax.config.update("jax_default_matmul_precision", "float32")` (or `"highest"`) before building RBM or running high-precision `symm-test` checks.
   - `qtx.set_default_dtype(jnp.float64)` does **not** override `quantax.model.*` dtypes (models intentionally pick `float32` for speed).
4. **Overflow-safe `LogArray` arithmetic**:
   - `state(s)` returns a `LogArray`. Always divide `LogArray`s first (`psi_conn / psi`) before calling `jnp.asarray(...)` to avoid fp32 underflow/overflow on larger lattices.
5. **Linear operator construction**:
   - When summing many operators, start from `op = qtx.operator.Operator([])` and use `op += term` in-place (never `sum(ops)`, which is quadratic).
6. **Randomness**:
   - Call `qtx.set_random_seed(seed)` at startup. Never call `qtx.get_subkeys()` inside `@jax.jit`.

## 5. Rigorous Physics Verification Workflow

Never report a raw single-step VMC energy without validation. Combine the absorbed companion skills:

1. **Exact reference (when $N \le 20$)**: Compare against QuSpin ED (`H.diagonalize(full_symm)`) and compute both exact variational energy `dense @ H @ dense` and ground-state fidelity `|dense @ wf[:, 0]|**2`.
2. **Symmetry check**: Run [symm-test](../symm-test/SKILL.md) to verify architecture-enforced symmetries ($O_T = 1.0000$) and measure candidate projection overlaps.
3. **Production measurement with autocorrelation binning**: Run [measure](../measure/SKILL.md) (`BinningAccumulator`) on thermalized chains to obtain converged error bars, autocorrelation time $\tau$, and check for frozen chains (`samples.r == 0` or zero sweep-to-sweep scatter).
4. **Zero-variance extrapolation**: Record `[step, energy, VarE]` every optimization step to a `.txt` log and run [zero-var-extrap](../zero-var-extrap/SKILL.md) (`odrpack` ODR fit + window grid + lever arm $R$ + variational-bound check).
5. **Performance & VRAM tuning (before large runs)**: Run [time-bench](../time-bench/SKILL.md) to pick optimal `max_parallel` and check Jacobian/SR memory headroom.

## 6. Google Colab Pro+ & W&B Integration Rules (This Repository)

- **Environment**: Colab 2026 ships Python 3.13 + JAX 0.11+ (CUDA 13 native). Install with:
  ```bash
  pip install -q "quantax[full]" odrpack wandb pyyaml matplotlib
  ```
  - **NEVER** install `jax-cuda12-plugin` or `netket[cuda]` alongside Colab's native CUDA 13 JAX.
  - Check `jax.devices()` at startup to verify GPU backend is active.
- **Checkpoints & Chains**: Save both `.eqx` parameters (`state.save(...)`) and sampler spin chains (`qtx.utils.to_replicated_numpy(sampler._spins)`) plus full step-by-step `[step, energy, VarE]` logs so W&B Artifacts and post-hoc `zero-var-extrap` work seamlessly.
- **Timeout & Compute Units**: Always pass `--timeout 0` to `colab exec` / `colab run`, and immediately run `colab_quick.sh stop` after interactive verification sessions.
