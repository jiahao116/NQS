# Research log

## 2026-11-22
- 选题：mixD 三层 vs 双层 t∥-J∥-J⊥ 的 NQS 受控比较（La₄Ni₃O₁₀ 层间配对受挫）。方案见 `proposals/trilayer_mixd_nqs.md`。
- 查新：未检索到三层 mixD + NQS 的工作；记录见 `notes/literature/novelty_check_2026-11-22.md`。未直接查询 arXiv API，需手动复核。
- 建立 `research/` 工作区。
- 下一步：手动复核 arXiv 查新链接 → 阶段 0（三层半满 Heisenberg + QuSpin ED 校验）。

## 2026-11-22（阶段 0：半满 ED）
- 新增 `src/mixd/`：`lattice.py`（多层格点与去重键表，唯一约定来源）、`ed_numba.py`（无 QuSpin 的矩阵无关 ED，Lin 表 O(1) 查找）、`ed_quspin.py`（QuSpin 交叉校验，待安装 QuSpin）、`quantax_ops.py`（NQS 用算符，未执行）。
- 配置：`configs/stage0_ed_scan.yaml`；NQS 配置 `stage0_{bilayer_4x2,trilayer_4x2,trilayer_4x4}_half.yaml`（runner 未写）。
- 运行 `scripts/run_ed_stage0.py`：4x2x3 三层（24 格点，dim 2,704,156）每点约 25 s。结果在 `ed/stage0/`（每点一个 JSON + `summary.md`）。
- **全部解析校验通过**：4x4 单层 E0=-11.22848321（文献 -0.70178020×16，差 8e-9）；4x2x2(J⊥=J∥) 与 4x4 环面图同构、能量一致；J⊥=0 退化为 n_L 份单层；J∥=0 柱子解析值（双层 -3/4 J⊥、三层 -J⊥ 每柱）；Σ J_b⟨S·S⟩_b = E0；基态为单态。
- 物理初看（仅半满，Ly=2 准一维团簇）：J⊥/J∥=4 时双层层间 ⟨S·S⟩=-0.704（接近单态 -0.75）、自旋能隙 2.62；三层层间仅 -0.477（上限 -0.5），能隙 0.71，内层面内关联被显著削弱（-0.143 vs 外层 -0.250）。与"内层被两侧争夺"的受挫图像一致（奇数层柱子带净自旋 1/2）。
- 发现：公开仓库 README 的 4x4 Heisenberg QuSpin 参考值 -44.91392517（σ·σ）比真值 -44.91393283 高 7.7e-6；据此 README 中 NQS E=-44.913952 实际低于真基态 1.9e-5（float32 量级），"相对误差 5.95e-07" 的表述需修正（待处理）。
- 下一步：安装 QuSpin 做第二条 ED 路径交叉校验；写 research NQS runner，先跑 4x2x{2,3} 对 ED。

## 2026-11-23（阶段 0：Colab 上的 NQS 对 ED）
- `research/` 改为随公开仓库发布；README 删除 float32 参考值导致的 4x4 Heisenberg 基准声明（quantax `Operator.diagonalize` 默认 float32）。
- 新增 `src/mixd/train_quantax.py`（ResConv + SR 两阶段，Stage 2 投影 SpinInverse；float64 QuSpin 算符校验与稠密评估）；`scripts/colab_remote_runner.py` 支持配置中的 `runner:` 键并把 `research/src` 加入 PYTHONPATH。
- Colab T4（热会话，运行后已释放）结果，同一 ansatz（ResConv 2×8，6,952 参数）、同一步数（300+300）：

| 团簇 | 算符校验（vs numba ED） | E_exact | E_NQS（稠密） | 相对误差 | 保真度 | 训练时间 |
| :-- | :-- | --: | --: | --: | --: | --: |
| 4x2x2 双层 | PASS（差 9e-15） | -11.2284832 | -11.2282753 | 1.85e-5 | 0.999957 | 125 s |
| 4x2x3 三层 | PASS（差 1e-14） | -17.5353163 | -17.5319170 | 1.94e-4 | 0.998825 | 169 s |

- 观察：同样设置下三层比双层难约一个数量级（三层能隙更小、内层受挫），且 Stage 2 末尾 VarE 仍在下降，尚未收敛。原始日志与 JSON 在 `results/stage0_nqs/`。
- 下一步：三层加长训练 / 加宽网络看能否到 ~1e-5；然后上 4x4x{2,3}（无 ED，可用 SSE-QMC 或尺寸外推作参考）；准备阶段 1（少空穴 t-J）。
