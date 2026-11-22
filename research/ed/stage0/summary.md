# Stage 0 ED: half-filled mixD multilayer Heisenberg

Convention: H = J_par sum_inplane S.S + J_perp sum_interlayer S.S (spin-1/2). Clusters periodic in-plane (bonds deduplicated for length-2 directions), open between layers. Backend: matrix-free numba ED (research/src/mixd/ed_numba.py).

| cluster | J_perp/J_par | dim(Sz=0) | E0 | E0/N | spin gap | <S.S> in-plane (per layer) | <S.S> interlayer | m_stag^2 |
| :-- | --: | --: | --: | --: | --: | :-- | :-- | --: |
| 4x2x1_pp | 0 | 70 | -4.82008937 | -0.60251117 | 0.82009 | L0:-0.4017 | - | 0.35375 |
| 4x4x1_pp | 0 | 12870 | -11.22848321 | -0.70178020 | 0.57860 | L0:-0.3509 | - | 0.27653 |
| 4x2x2_pp | 0 | 12870 | -9.64017875 | -0.60251117 | 0.82009 | L0:-0.4017, L1:-0.4017 | L01:0.0000 | 0.17688 |
| 4x2x2_pp | 0.5 | 12870 | -10.08474665 | -0.63029667 | 0.56824 | L0:-0.3848, L1:-0.3848 | L01:-0.2124 | 0.25725 |
| 4x2x2_pp | 1 | 12870 | -11.22848321 | -0.70178020 | 0.57860 | L0:-0.3509, L1:-0.3509 | L01:-0.3509 | 0.27653 |
| 4x2x2_pp | 2 | 12870 | -14.88921007 | -0.93057563 | 0.93838 | L0:-0.2520, L1:-0.2520 | L01:-0.5525 | 0.23537 |
| 4x2x2_pp | 4 | 12870 | -25.29422829 | -1.58088927 | 2.61501 | L0:-0.1158, L1:-0.1158 | L01:-0.7036 | 0.15043 |
| 4x2x2_pp | inf | 12870 | -6.00000000 | -0.37500000 | 1.00000 | L0:0.0000, L1:0.0000 | L01:-0.7500 | 0.09375 |
| 4x2x3_pp | 0 | 2704156 | -14.46026812 | -0.60251117 | 0.82009 | L0:-0.4017, L1:-0.4017, L2:-0.4017 | L01:0.0000, L12:-0.0000 | 0.11792 |
| 4x2x3_pp | 0.5 | 2704156 | -15.35239165 | -0.63968299 | 0.45293 | L0:-0.3842, L1:-0.3704, L2:-0.3842 | L01:-0.2110, L12:-0.2110 | 0.21297 |
| 4x2x3_pp | 1 | 2704156 | -17.53531626 | -0.73063818 | 0.43636 | L0:-0.3553, L1:-0.3212, L2:-0.3553 | L01:-0.3221, L12:-0.3221 | 0.24295 |
| 4x2x3_pp | 2 | 2704156 | -23.61984140 | -0.98416006 | 0.55138 | L0:-0.3046, L1:-0.2330, L2:-0.3046 | L01:-0.4223, L12:-0.4223 | 0.23273 |
| 4x2x3_pp | 4 | 2704156 | -38.25556744 | -1.59398198 | 0.70734 | L0:-0.2496, L1:-0.1427, L2:-0.2496 | L01:-0.4774, L12:-0.4774 | 0.19818 |
| 4x2x3_pp | inf | 2704156 | -8.00000000 | -0.33333333 | -0.00000 | L0:-0.0172, L1:-0.0043, L2:-0.0172 | L01:-0.5000, L12:-0.5000 | 0.06948 |

## Analytic-limit checks

- PASS: 4x4 single layer E0=-11.22848321 vs literature -11.22848320 (diff -8.4e-09); repo README value -11.22848129 differs by -1.9e-06
- PASS: graph isomorphism: 4x2x2 (J_perp=J_par, Ly=2 deduplicated) == 4x4 torus (diff -3.6e-15)
- PASS: J_perp=0: E(4x2x2) = 2 E(4x2x1) (diff -1.1e-14)
- PASS: J_par=0: 2-site columns E0=-6.00000000 vs analytic -6.0 (diff -8.9e-16)
- PASS: J_perp=0: E(4x2x3) = 3 E(4x2x1) (diff -5.3e-15)
- PASS: J_par=0: 3-site columns E0=-8.00000000 vs analytic -8.0 (diff 0.0e+00)
- PASS: 4x2x1_pp_jpar1_jperp0: sum_b J_b <S.S>_b = E0 (diff -3.6e-15)
- PASS: 4x2x1_pp_jpar1_jperp0: singlet ground state, <S_tot^2>=-2.4e-15
- PASS: 4x4x1_pp_jpar1_jperp0: sum_b J_b <S.S>_b = E0 (diff 8.3e-14)
- PASS: 4x4x1_pp_jpar1_jperp0: singlet ground state, <S_tot^2>=1.3e-13
- PASS: 4x2x2_pp_jpar1_jperp0: sum_b J_b <S.S>_b = E0 (diff 1.1e-14)
- PASS: 4x2x2_pp_jpar1_jperp0: singlet ground state, <S_tot^2>=1.7e-13
- PASS: 4x2x2_pp_jpar1_jperp0.5: sum_b J_b <S.S>_b = E0 (diff -1.2e-14)
- PASS: 4x2x2_pp_jpar1_jperp0.5: singlet ground state, <S_tot^2>=1.3e-13
- PASS: 4x2x2_pp_jpar1_jperp1: sum_b J_b <S.S>_b = E0 (diff 2.7e-14)
- PASS: 4x2x2_pp_jpar1_jperp1: singlet ground state, <S_tot^2>=1.7e-14
- PASS: 4x2x2_pp_jpar1_jperp2: sum_b J_b <S.S>_b = E0 (diff 7.1e-15)
- PASS: 4x2x2_pp_jpar1_jperp2: singlet ground state, <S_tot^2>=-7.2e-14
- PASS: 4x2x2_pp_jpar1_jperp4: sum_b J_b <S.S>_b = E0 (diff 3.6e-13)
- PASS: 4x2x2_pp_jpar1_jperp4: singlet ground state, <S_tot^2>=2.3e-13
- PASS: 4x2x2_pp_jpar0_jperp1: sum_b J_b <S.S>_b = E0 (diff 8.9e-16)
- PASS: 4x2x2_pp_jpar0_jperp1: singlet ground state, <S_tot^2>=3.3e-17
- PASS: 4x2x3_pp_jpar1_jperp0: sum_b J_b <S.S>_b = E0 (diff -5.7e-12)
- PASS: 4x2x3_pp_jpar1_jperp0: singlet ground state, <S_tot^2>=-1.9e-11
- PASS: 4x2x3_pp_jpar1_jperp0.5: sum_b J_b <S.S>_b = E0 (diff 5.5e-13)
- PASS: 4x2x3_pp_jpar1_jperp0.5: singlet ground state, <S_tot^2>=-1.6e-12
- PASS: 4x2x3_pp_jpar1_jperp1: sum_b J_b <S.S>_b = E0 (diff -1.2e-13)
- PASS: 4x2x3_pp_jpar1_jperp1: singlet ground state, <S_tot^2>=6.1e-13
- PASS: 4x2x3_pp_jpar1_jperp2: sum_b J_b <S.S>_b = E0 (diff -7.9e-13)
- PASS: 4x2x3_pp_jpar1_jperp2: singlet ground state, <S_tot^2>=-2.8e-12
- PASS: 4x2x3_pp_jpar1_jperp4: sum_b J_b <S.S>_b = E0 (diff -6.6e-13)
- PASS: 4x2x3_pp_jpar1_jperp4: singlet ground state, <S_tot^2>=-1.7e-12
- PASS: 4x2x3_pp_jpar0_jperp1: sum_b J_b <S.S>_b = E0 (diff -2.6e-12)
