# G-FNO Paddle 动转静 + CINN 训练加速：实现与基准报告

> 日期：2026-07-29　|　范围：仅 `GFNO2d_p4`（reflection=False）主线　|　环境：RTX 4060 Ti 16GB、PaddlePaddle 3.3.0（CUDA）、conda env `paddletorch`

## 一、结论

CINN 动转静训练链路已打通：单一开关 `GFNO_USE_CINN=1` 即启用，**动态图行为完全不变（默认关）**，CINN 真正编译训练图，且与动态图数值逐位一致、G-等变性保持。

**纯训练稳态加速比**（forward+backward+optimizer step，已预热、排除数据加载与一次性编译）：

| 分辨率 | DYN (ms/步) | CINN (ms/步) | 纯训练加速 |
|---|---|---|---|
| 16×16（小型 SWE，modes=8） | 39.17 | 36.21 | **7.6%** |
| 32×32（原版 128 下采样，modes=12） | 42.15 | 37.26 | **11.6%** |

**端到端**（含 CINN 一次性编译 ~100s）：短跑下 CINN 反而更慢；收益需长训练才能显现。

| 分辨率 | CINN 编译(s) | 每步节省(ms) | 纯训练 break-even(步) |
|---|---|---|---|
| 16×16 | ~100 | 2.96 | ~33,700 |
| 32×32 | ~106 | 4.89 | ~21,700 |

> 与《CINN 动转静训练踩坑手册》一致：CINN 有 ~100s 编译 warmup，短跑总耗时反比动态图长，**看加速比只看稳态步速**；长训练（>break-even 步数）才有净收益。

## 二、实现方案（B1+）

阻塞与方案详见 `docs/cinn_rot90_investigation.md`。此处仅述落地改动。

- **单一开关**（`G-FNO_paddle/experiments.py`）：读 `GFNO_USE_CINN` 环境变量；开则在 `import paddle` 前 `setdefault` 三个 FLAGS，并在权重加载后 `paddle.jit.to_static(model, full_graph=False)`；关则原版纯动态。
- **用 SOT（`full_graph=False`）**：GFNO2d 等变权重组装会在前向内重赋值 `self.weights`，AST（`full_graph=True`）禁止；SOT 保留该动态语义且同样触发 CINN。
- **Hermitian 路径实/虚分量组装**（`models/GFNO.py` `GConv2d.get_weight`，`reflection=False` 分支）：在 `float32` 实/虚分量上完成 Hermitian 拼接、共轭（虚部取负）、空间旋转、群通道移位，末尾一次 `paddle.complex(real, imag)` 恢复，规避 CINN 不支持的 `complex64 concat`。`reflection=True`（p4m）保留原 complex 路径不破坏动态图。
- **`rot90` 替换**：`paddle.rot90` 不支持 `complex64` 且 SOT 无法模拟；全部改为数值验证过的 `flip + transpose`（`_rot90_safe`）。
- **机械兼容改造**（2D 路径）：`.view()→.reshape([...])`、`.size()→.shape[]`、`.repeat([...])→paddle.tile(...)`、`GSpectralConv2d` 的 `.nonzero().item()`→静态 `freq0_y = x.shape[-2]//2`。
- **`paddle.zeros` 修复**：`GSpectralConv2d` 的 `out_ft = paddle.zeros(..., device=x.device)` 在 32×32 CINN 下报 "multiple values for argument 'dtype'"；改为列表 shape + 去掉 `device=`（paddle.zeros 用全局设备，数值等价）。

## 三、数值等价验证（验收 #1/#2）

- **动态图逐位等价**（移植 B1+ 前后，modes=8/width=10/16×16）：输出 `max_abs_diff=0.0`，全部 42 个梯度 `max_abs_diff=0.0`。
- **真实训练管线 DYN vs CINN loss 一致**：
  - 16×16：train 0.14856 / valid 0.04827 / test 0.04647（两侧相同至 5 位）
  - 32×32：train 0.15677 / valid 0.05342 / test 0.04987（两侧相同至 5 位）
- **G-等变性保持**：`Rotations` 指标 CINN on/off 均为 ~1e-7（≈0，严格等变），`Reflections`（p4 无反射，预期非零）一致。

## 四、跑通矩阵（验收 #3/#4，单卡）

| 数据集 | 分辨率 | CINN 关(DYN) | CINN 开 |
|---|---|---|---|
| 小型 SWE（`2D_rdb_NA_NA.h5`，200 轨 64 原） | 16×16 | ✅ 训练 | ✅ 训练（CINN 编译） |
| 原版 PDEBench SWE（128 原，80 轨子集） | 32×32 | ✅ 训练 | ✅ 训练（CINN 编译） |

"跑通"= 1 epoch 正常训练、loss/梯度有限、日志出现 `Compiling subgraph with CINN backend`。

> 原版全量（1000 轨，6.6GB）加载峰值约 13GB，超出本机可用内存（~9.6GB），故用原生 128 分辨率的 80 轨子集验证（rdb 加载分支无硬编码分辨率，格式与全量一致，参数 `--rdb_super_res=128 --rdb_downsample=4 --modes=12`）。

## 五、改动文件清单

- `G-FNO_paddle/experiments.py`：`GFNO_USE_CINN` 单一开关 + `to_static(full_graph=False)`。
- `G-FNO_paddle/models/GFNO.py`：新增 `_rot90_safe`/`_rot90_real_once`/`_hermitian_real_imag`；`GConv2d.get_weight` B1+ 改写（仅 `reflection=False` 主线走新路径）；`GSpectralConv2d.forward` 静态频率索引 + `.size→.shape` + `paddle.zeros` 修复；`GNorm`/`GFNO2d.forward` `.view→.reshape`。**未改动 3D 路径与其他 model_type。**
- `G-FNO_paddle/utils.py`：`grid.twoD_grid` `.repeat→paddle.tile`（仅 2D）。

## 六、使用方法

```bash
# 动态图（默认，行为不变）
cd G-FNO_paddle && python experiments.py --seed=1 --data_path=... --model_type=GFNO2d_p4 ...

# CINN 动转静训练
GFNO_USE_CINN=1 python experiments.py --seed=1 --data_path=... --model_type=GFNO2d_p4 ...
```

初始化复现：测试均用 `--seed=1`（`experiments.py` 内 `paddle.seed`），CINN on/off 同 seed 即同初始化，无需额外权重文件。若需持久化 paddle 随机权重，`paddle_random_init/`（仅 paddle 依赖）可 `git checkout HEAD -- paddle_random_init/` 恢复后运行 `generate_weights.py`。

## 七、方法论与限制

- **纯训练测度**：受控脚本（`/tmp/bench_cinn.py`，合成 batch 排除数据加载），20 步预热后 3 段 ×50 步取均值；DYN/CINN 同 seed、同模型。
- **端到端**含一次性 CINN 编译（~100s），短跑下 CINN 总耗时更长；break-even 约 2.2–3.4 万步。
- SOT（`full_graph=False`）下 CINN 覆盖度低于 AST，加速比有限（7.6%–11.6%）；更大分辨率/批次收益更高。
- 仅 `GFNO2d_p4`（reflection=False）经验证；`p4m` 及 3D 未纳入（保留原动态路径）。
