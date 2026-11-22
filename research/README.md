# research/ — Mixed-dimensional multilayer t-J study with neural quantum states

Work-in-progress research workspace. The main toolkit (`src/nqs`, `configs/`, `scripts/`)
stays generic; everything specific to this study lives here.

## Current project
**Controlled NQS comparison of mixed-dimensional (mixD) bilayer vs trilayer
t∥-J∥-J⊥ models — a test of the "interlayer pairing frustration" scenario for the
trilayer nickelate La₄Ni₃O₁₀.** See [proposals/trilayer_mixd_nqs.md](proposals/trilayer_mixd_nqs.md)
(in Chinese) and the running log [notes/log.md](notes/log.md).

## Layout
| Path | Content |
| :--- | :--- |
| `proposals/` | Project plan and staged milestones |
| `notes/` | Research log `log.md`; `literature/` holds literature and novelty checks |
| `configs/` | Experiment YAML configs (`backend: quantax`) |
| `src/mixd/` | Study-specific code: multilayer lattice / bond lists, numba ED, QuSpin ED, quantax operators |
| `scripts/` | Batch and analysis scripts (e.g. `run_ed_stage0.py`) |
| `ed/` | Exact-diagonalization reference results |
| `runs/` | Training outputs and checkpoints (git-ignored) |
| `figures/` | Figures |

## Conventions
- Code, comments, logs and configs are in English; notes and proposals may be in Chinese.
- Spin models use spin-1/2 operators (S·S). `quantax.operator.Heisenberg` uses σ·σ = 4 S·S.
- All Hamiltonian backends build from the single bond list in `src/mixd/lattice.py`.
- Append an entry to `notes/log.md` for every substantive step (date, what, result, next).
- Never store credentials here.

## Reproduce Stage 0 (half-filled ED)
```bash
cd research
python scripts/run_ed_stage0.py --config configs/stage0_ed_scan.yaml --max-sites 24
```
Requires `numpy`, `scipy`, `numba`, `pyyaml`. Results go to `ed/stage0/` (`summary.md` + one JSON per point).
