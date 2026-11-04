# pk_ring_anyangle 三项指标说明

方案：**`pk_ring_anyangle`**（circ-harmonic NTRU 全角度，仅改 z₀，检测 = key + 图 + VAE，无搜角 / 无 deskew）。

---

## 1. 检测率（TPR，有水印能否检出）

**定义（与大样本脚本一致）：** 在 **未旋转** 的 SD 全流程水印图上，`peak ≥ 阈值` 的比例。  
**不算** 旋转图进 F1 的 TP/FN；旋转单独统计见下表「旋转检测率」。

| 项目 | 数值 |
|------|------|
| **干净水印检测率** | **85.4%**（**41 / 48**） |
| 阈值 | 0.5（脚本：`max(0.5, q85(负样本 peak)+0.03)`） |
| Prompt / Steps | `a photo of a beautiful woman`，15 |
| 数据来源 | `rotation_wm/output/anyangle_largesample_20260922_195201/summary.json` |

**补充（全角度攻击，非「误报」口径）：**

| 项目 | 数值 |
|------|------|
| 旋转检测率（expand+resize，Unif[0,360°)，每 seed 2 角） | **46.9%**（**45 / 96**） |
| 召回 Recall（仅 clean 正类，同次 run） | **85.4%**（与干净水印检测率相同） |
| F1（clean 正类 + 8 张无水印负类） | **0.911** |

**一句话：** 不旋转时约 **85%** 能检出；随机全角度旋转后约 **47%** 仍能过阈。

---

## 2. 误报率（无水印被当成有水印）

**定义：** 无水印 SD 负样本中，`peak ≥ 阈值` 的比例  
\(\mathrm{FPR} = \mathrm{FP} / (\mathrm{FP}+\mathrm{TN})\)。

| 项目 | 数值 |
|------|------|
| 负样本数 | 8 |
| 误报 FP | 1 |
| 正确拒绝 TN | 7 |
| **误报率** | **12.5%**（**1 / 8**） |
| Precision | 97.6%（41 / (41+1)） |

该次误报 peak 约 **0.93**（其余 7 张负样本 peak 多在 0–0.31）。  
数据来源：同上 `summary.json` 中 `neg_peaks`、`fp`、`tn`。

---

## 3. 图像质量

大样本评测协议声明：**目视 PNG**，**未** 报 SSIM/PSNR（见 `summary.json` → `protocol.quality`）。  
下列为仓库内 **额外跑过** 的质量相关数字，便于量化对比。

### 3.1 目视（主协议）

| 项目 | 说明 |
|------|------|
| 样本 | 10 个 seed 的 clean / wm 条带图 |
| 路径 | `rotation_wm/output/anyangle_largesample_20260922_195201/visual/`、`gallery.html` |

### 3.2 NIQE（无参考，**越低越好**；相对自然图统计）

小样本：**4 seed**，同 prompt/steps，与无水印 SD 对照。

| 方案 | NIQE 均值 ± 标准差 |
|------|-------------------|
| 无水印 clean | **5.64 ± 0.31** |
| **any-angle 水印** | **5.86 ± 0.86** |

数据来源：`rotation_wm/output/niqe_harm_b3_20260923_112230/summary.json` → `b3_anyangle`。

### 3.3 FID（相对真实图 / 相对 clean SD）

参考集：**Oxford-IIIT Pet test，200 张**（真实宠物照片；与「人像」prompt **域不一致**，绝对 FID 仅作参考）。

| 对比 | FID（越低越接近对比集分布） |
|------|----------------------------|
| clean SD → 宠物参考 | 318.50 |
| any-angle wm → 宠物参考 | 318.21 |
| **Δ（wm − clean）** | **−0.29** |
| **clean SD ↔ any-angle wm**（同 pipeline 两堆生成图） | **102.71** |

样本：12 seed 目标，目录内各 24 PNG（含一次中断重跑的重复 seed，配对 FID 为粗估）。  
数据来源：`rotation_wm/output/fid_realref_20261011_231025/fid_summary.json`。

**质量小结：** 相对无水印 SD，NIQE 略升（4 seed）；对宠物参考 FID 几乎不变；**水印造成的成对分布差 FID ≈ 103**（样本有限）。主结论仍建议结合 **gallery 目视**。

---

## 4. 复现命令

```bash
# 检测率 / 误报率（48 seed + 旋转 + 8 neg）
python -m rotation_wm._eval_anyangle_largesample

# NIQE（4 seed，多方案对比）
python -m rotation_wm._eval_niqe_harm_b3

# FID（需生成 + 参考集下载）
python -m rotation_wm._eval_fid_realref --n-seeds 12 --max-ref 200
```

---

## 5. 汇总表（直接写论文/报告时可引用）

| 指标 | 数值 | 备注 |
|------|------|------|
| **检测率（clean 水印）** | **85.4%** | 41/48，阈值 0.5 |
| **旋转后检测率** | **46.9%** | 45/96，全角度 expand+resize |
| **误报率** | **12.5%** | 1/8 无水印 SD |
| **图像质量（NIQE，4 seed）** | **5.86** vs clean **5.64** | 越低越好，略差 |
| **图像质量（FID clean↔wm）** | **102.7** | 12 seed 粗估；越低越好 |
| **PR（环外 uniform-bits）** | **0.992** | 嵌入约束，≥0.99 门槛 |
