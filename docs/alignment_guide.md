# G-FNO Torch/Paddle 训练 Loss 对齐指南

## 1. 背景

`GFNO2d_p4` 模型的 Paddle (PaddleCFD) 和 Torch (G-FNO) 版本在默认训练中 loss 不对齐，根因有三：

| # | 根因 | 修复方式 |
|---|------|---------|
| 1 | Paddle `GNorm` 多 20 个可训练 affine 参数 | 已修复：`weight_attr=False, bias_attr=False` |
| 2 | 训练入口未加载共享初始权重 | opt-in 参数 `--debug_initial_state_path` |
| 3 | 双侧 DataLoader shuffle 顺序不同 | opt-in 参数 `--debug_batch_order_path` + `DebugEpochBatchLoader` |

## 2. 快速开始：生成共享输入

```bash
cd /home/lkyu/baidu/G-FNO
conda run -n paddletorch python tools/debug_loss_alignment/make_alignment_inputs.py \
  --out-dir "/tmp/gfno_alignment" \
  --seed 1 \
  --ntrain 160 --t-in 1 --t-out 24 \
  --batch-size 128 --epochs 10
```

产出：
- `/tmp/gfno_alignment/gfno2d_p4_torch_initial.pt` — Torch 初始权重
- `/tmp/gfno_alignment/gfno2d_p4_paddle_initial.pdparams` — Paddle 初始权重（由 Torch 转换）
- `/tmp/gfno_alignment/torch_epoch_batches.json` — 多 epoch 共享 batch 顺序

## 3. Paddle 侧受控训练 (PaddleCFD)

PaddleCFD (`/home/lkyu/baidu/PaddleCFD`) 的 `feat/G-FNO` 分支已集成 debug 参数。

```bash
cd /home/lkyu/baidu/PaddleCFD/examples/G-FNO
conda run -n paddletorch python experiments.py \
  --seed=1 \
  --data_path="/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5" \
  --results_path="/tmp/gfno_alignment/paddle_results" \
  --strategy=teacher_forcing \
  --T=24 --ntrain=160 --nvalid=20 --ntest=20 \
  --model_type=GFNO2d_p4 --modes=8 --width=10 \
  --batch_size=128 --epochs=10 --learning_rate=1e-3 \
  --early_stopping=20 --verbose \
  --rdb_super_res=64 --rdb_downsample=4 \
  --debug_initial_state_path="/tmp/gfno_alignment/gfno2d_p4_paddle_initial.pdparams" \
  --debug_batch_order_path="/tmp/gfno_alignment/torch_epoch_batches.json"
```

## 4. Torch 侧受控训练 (G-FNO)

G-FNO 项目 (`/home/lkyu/baidu/G-FNO`) 中的 `G-FNO/experiments.py` 需要做以下修改才能与 Paddle 侧进行受控训练对齐。

### 4.1 添加 Parser 参数

在 `G-FNO/experiments.py` 的 `parser.add_argument` 区域末尾（`--rdb_downsample` 之后）添加：

```python
parser.add_argument("--debug_initial_state_path", type=str, default=None, help="debug-only path to a Torch state_dict")
parser.add_argument("--debug_batch_order_path", type=str, default=None, help="debug-only JSON file with epoch_batches")
```

### 4.2 模型构造后加载 debug 初始权重

在 model construction 结束处（`raise NotImplementedError("Model not recognized")` 之后）添加：

```python
if args.debug_initial_state_path is not None:
    model.load_state_dict(torch.load(args.debug_initial_state_path, map_location="cuda"))
```

### 4.3 DataLoader 后加载 debug batch order

在 `train_loader = torch.utils.data.DataLoader(...)` 之后、optimizer setup 之前添加：

```python
debug_epoch_batches = None
if args.debug_batch_order_path is not None:
    from debug_alignment import DebugEpochBatchLoader, load_debug_epoch_batches

    debug_epoch_batches = load_debug_epoch_batches(args.debug_batch_order_path)
    assert len(debug_epoch_batches) >= epochs, "debug batch order must cover all epochs"
    train_loader = DebugEpochBatchLoader(train_data, debug_epoch_batches[0])
```

### 4.4 Epoch 循环中切换 batch order

在 `for ep in range(epochs):` 之后添加：

```python
    if debug_epoch_batches is not None:
        train_loader = DebugEpochBatchLoader(train_data, debug_epoch_batches[ep])
```

### 4.5 需要 G-FNO/debug_alignment.py

Torch 侧还需要 `G-FNO/debug_alignment.py` 文件（已在 G-FNO 项目中存在）：

```python
from __future__ import annotations

from pathlib import Path
import json

import torch


def load_debug_epoch_batches(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        [[int(index) for index in batch] for batch in epoch]
        for epoch in payload["epoch_batches"]
    ]


class DebugEpochBatchLoader:
    def __init__(self, dataset, epoch_batches):
        self.dataset = dataset
        self.epoch_batches = [[int(index) for index in batch] for batch in epoch_batches]

    def __len__(self):
        return len(self.epoch_batches)

    def __iter__(self):
        for batch_indices in self.epoch_batches:
            xs = []
            ys = []
            for index in batch_indices:
                x, y = self.dataset[index]
                xs.append(x)
                ys.append(y)
            yield torch.stack(xs, dim=0), torch.stack(ys, dim=0)
```

### 4.6 Torch 受控训练命令

修改完成后运行：

```bash
cd /home/lkyu/baidu/G-FNO/G-FNO
conda run -n paddletorch python experiments.py \
  --seed=1 \
  --data_path="/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5" \
  --results_path="/tmp/gfno_alignment/torch_results" \
  --strategy=teacher_forcing \
  --T=24 --ntrain=160 --nvalid=20 --ntest=20 \
  --model_type=GFNO2d_p4 --modes=8 --width=10 \
  --batch_size=128 --epochs=10 --learning_rate=1e-3 \
  --early_stopping=20 --verbose \
  --rdb_super_res=64 --rdb_downsample=4 \
  --debug_initial_state_path="/tmp/gfno_alignment/gfno2d_p4_torch_initial.pt" \
  --debug_batch_order_path="/tmp/gfno_alignment/torch_epoch_batches.json"
```

## 5. Loss 对齐结果

使用 `test/compare_forward_diff.py` 工具脚本对比双侧训练日志：

```bash
conda run -n paddletorch python test/compare_forward_diff.py \
  --torch-log "/tmp/gfno_alignment/torch_controlled.log" \
  --paddle-log "/tmp/gfno_alignment/paddle_controlled.log"
```

输出：

```
-------------------------------------------------------------------------------------------------------------------
G-FNO Controlled Training Loss Comparison
-------------------------------------------------------------------------------------------------------------------
Epoch |      Torch Train |     Paddle Train |     Train Diff |      Torch Valid |     Paddle Valid |     Valid Diff
-------------------------------------------------------------------------------------------------------------------
    0 |   0.281261720570 |   0.281335672177 |   7.395161e-05 |   0.085092836618 |   0.085185599327 |   9.276271e-05
    1 |   0.018453580979 |   0.018482862413 |   2.928143e-05 |   0.067927408218 |   0.066261696815 |   1.665711e-03
    2 |   0.009400227868 |   0.009394724481 |   5.503387e-06 |   0.038629686832 |   0.043152570724 |   4.522884e-03
    3 |   0.006490346914 |   0.006522691514 |   3.234460e-05 |   0.041562762856 |   0.078496730328 |   3.693397e-02
    4 |   0.005093538621 |   0.004809832565 |   2.837061e-04 |   0.034986469150 |   0.069000387192 |   3.401392e-02
    5 |   0.005873561472 |   0.004971065659 |   9.024958e-04 |   0.028754043579 |   0.019240155816 |   9.513888e-03
    6 |   0.003246136390 |   0.004314367046 |   1.068231e-03 |   0.042773655057 |   0.029351681471 |   1.342197e-02
    7 |   0.002722640871 |   0.002368335577 |   3.543053e-04 |   0.016586881876 |   0.015984503925 |   6.023780e-04
    8 |   0.001788152358 |   0.001759155809 |   2.899655e-05 |   0.019331794977 |   0.016444309056 |   2.887486e-03
    9 |   0.001650591346 |   0.001652324215 |   1.732869e-06 |   0.014188805223 |   0.013227781653 |   9.610236e-04
-------------------------------------------------------------------------------------------------------------------
 Test |   0.013804441690 |   0.013143031299 |   6.614104e-04
-------------------------------------------------------------------------------------------------------------------

Acceptance Criteria Check:
  Epoch 0 train diff 7.395161e-05 <= 1e-4: PASS
  Epoch 0 valid diff 9.276271e-05 <= 1e-4: PASS
  Epoch 9 train diff 1.732869e-06 <= 1e-3: PASS
  Epoch 9 valid diff 9.610236e-04 <= 1e-3: PASS
  Test loss diff  6.614104e-04 <= 1e-3: PASS
  Overall: ALL PASS
```

### 对比默认（未对齐）训练

| Backend | Epoch 0 train | Epoch 0 valid | Epoch 9 train | Epoch 9 valid | Test |
|---------|--------------:|--------------:|--------------:|--------------:|-----:|
| Torch   | 0.2818511234 | 0.0940160334 | 0.0016741997 | 0.0090169132 | 0.0117879599 |
| Paddle  | 0.4209095549 | 0.1181925535 | 0.0023952951 | 0.0262326151 | 0.0228278518 |

默认训练因不同初始权重和不同 batch 顺序，Epoch 0 的 Paddle loss 比 Torch 高约 49%。

## 6. 关键文件索引

| 文件 | 位置 | 说明 |
|------|------|------|
| PaddleCFD experiments.py | `PaddleCFD/examples/G-FNO/experiments.py` | Paddle 训练入口（已集成 debug 参数） |
| PaddleCFD GFNO.py | `PaddleCFD/ppcfd/models/g_fno/GFNO.py` | GNorm 已修复 |
| PaddleCFD debug_alignment.py | `PaddleCFD/examples/G-FNO/debug_alignment.py` | Paddle 侧 debug batch loader |
| G-FNO experiments.py | `G-FNO/experiments.py` | Torch 训练入口（需手动添加 debug 参数，见 §4） |
| G-FNO debug_alignment.py | `G-FNO/debug_alignment.py` | Torch 侧 debug batch loader |
| make_alignment_inputs.py | `tools/debug_loss_alignment/make_alignment_inputs.py` | 生成共享权重和 batch 顺序 |
| compare_forward_diff.py | `test/compare_forward_diff.py` | 对比两个训练日志的工具脚本 |
| README.md | `tools/debug_loss_alignment/README.md` | 完整排查工作报告 |
