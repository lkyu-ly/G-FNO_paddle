# 交接文档：G-FNO Paddle 动转静 + CINN 训练加速

> 本文档为接续 agent 准备。**请先完整阅读本文档再动手**，不要重复已完成的工作。
>
> **最高优先级约束（项目所有者明确要求，必须遵守）：**
> - **敏捷开发**：本仓库已具备正常运行能力，**只解决本文档列出的明确提出的问题，禁止过度设计、禁止假想/预防性问题、禁止探索问题链之外的任务。**
> - **疑难暂停**：若你即将做"大量修改代码/逻辑或更改依赖"才能解决的疑难改动，**暂停并向项目所有者说明根因与可选方案，不要自行大改。**
> - 用中文（简体）交流。

---

## 1. 主线任务目标

为 `G-FNO_paddle`（PyTorch→PaddlePaddle 迁移版）引入 **CINN 动转静训练加速**，并交付：

1. 用**单一开关**控制 CINN（开 = 动转静 + CINN；关 = 原版纯动态，默认关、行为不变）。
2. CINN 开启后获得**实质性的训练加速**（不是只在环境变量层面"能跑"而不生效）。
3. 本地单卡跑通**两个数据集**（小型 SWE + 原版 PDEBench SWE），CINN 关闭与开启各跑通（"跑通"= 几个 step 后能正常训练）。
4. 给出**确定性的加速比数据**（考虑 CINN 一次性编译 warmup，见 §9 方法），写入仓库文件。
5. （附带）排查 git 两个未同步提交。

**只测试 `GFNO2d_p4` 这一个 model_type**（主线用例）。其他型号不在本次范围。

---

## 2. 关键背景

- **CINN 仅在动转静后生效**：仅设 `FLAGS_use_cinn=true` 不会编译；必须 `paddle.jit.to_static(...)` 包裹模型，CINN 才编译训练图。生效标志：日志出现 `add_cinn_pass.cc:334 Compiling subgraph with CINN backend`。
- **Paddle 3.3.0 下 to_static 与 CINN 强绑定**：to_static 会自动启用 CINN，无法独立关闭。因此用**单一开关**同时控制（开 = to_static + FLAGS 全开；关 = 不 to_static + FLAGS 全 false）。
- G-FNO **无官方预训练权重**。本次随机初始化即可（`paddle.seed(args.seed)` 已在 experiments.py 设置，同 seed 可复现初始化）。

---

## 3. 已完成（不要重做）

1. **单一开关已加**（`G-FNO_paddle/experiments.py`）：
   - 顶部 `import paddle` 之前：读 `GFNO_USE_CINN` 环境变量，开则 `setdefault` 三个 FLAGS。
   - 模型构造 + 权重加载（`if args.debug_initial_state_path ...`）之后：`if _GFNO_USE_CINN: model = paddle.jit.to_static(model, full_graph=True)`。
   - 默认关，动态图行为完全不变。
2. **2D 路径机械兼容改造已完成**（限 GFNO2d 2D 路径，未动 3D）：
   - `models/GFNO.py`：`.view()→.reshape([...])`（GConv2d.get_weight 两处、GNorm 两处、GFNO2d.forward 一处）；`.repeat(...)→paddle.tile(..., [...])`（get_weight 两处）；`GSpectralConv2d.forward` 中 `.nonzero().item()` → 静态 `freq0_y = x.shape[-2] // 2`，并把 `x.size(-2/-1) → x.shape[-2/-1]`。
   - `utils.py`：`grid.twoD_grid` 两处 `.repeat([...])` → `paddle.tile(...)`。
3. **动态图回归通过**：改造后动态图前向+反向正常（未破坏原行为）。
4. **复数 FFT 子图经探针证明 CINN 可编译**：`as_complex + rfft2 + 复数 einsum + 复数 setitem + irfft2` 的最小子图，`full_graph=True` 下 CINN 编译成功，前向+反向均通过。**复数 FFT 本身不是阻塞。** 探针在 `/tmp/probe_cinn_complex.py`（如还在）。
5. **数据集兼容性已确认**（无需改代码，仅参数差异）：见 §7。
6. **git 未同步已查清**：见 §8。

---

## 4. 未完成

- [ ] **解除动转静在 GFNO2d 的阻塞**（当前核心卡点，见 §5）。
- [ ] 单卡跑通两个数据集（CINN off & on）。
- [ ] 基准测试加速比并写入仓库报告文件。
- [ ] （可选）准备 paddle 随机权重文件便于测试（恢复 `paddle_random_init/` 或用 `--seed` 确定性）。

---

## 5. 当前阻塞问题（接续 agent 的核心任务）

动转静在 **GFNO2d 的等变权重组装**处被阻断。动态图完全正常，纯粹是 to_static/CINN 的限制。有两层阻塞，按出现顺序：

### 阻塞 A：前向内重赋值 `self.weights`（仅 `full_graph=True` 触发）
- 位置：`GConv2d.get_weight`（`models/GFNO.py`，搜索 `def get_weight`，2D 版在 GConv2d 内）。
- 现象：`full_graph=True` 时报
  `TypeError: In transformed code: assignment to parameter 'weights' should be of type Parameter or None, but got 'Value'`
- 根因：`get_weight` 在前向里反复执行 `self.weights = <tensor>`、`self.bias = <tensor>`（`self.weights` 是由可训练参数 `self.W`（Parameter/ParameterDict）每次前向重算出的普通 Tensor 属性，非注册 Parameter）。AST 动转静禁止在前向里重赋值此类属性。
- 注意：`self.weights[:, k] = ...` 是 `__setitem__`（索引赋值），不是 `__setattr__`，**不触发**本错误（探针已验证 setitem 在 CINN 下 OK）。

### 阻塞 B：复数 `Tensor.rot90()`（SOT 与 full_graph 都触发，是真正的硬阻断）
- 位置：`GConv2d.get_weight`，对复数权重 `self.weights` 调用 `.rot90(...)`。搜索 `grep -n 'rot90' G-FNO_paddle/models/GFNO.py`：2D 三处（约 132/140/159 行，用于 Hermitian/等变核的 90°/180° 旋转），3D 也有（不在本次范围）。
- 现象（SOT，`full_graph=False`）：
  ```
  TypeError: The data type of 'X' in rot90 must be ['float16','float32','float64','int32','int64','bool'], but received complex64.
  TypeError: 'rot90' is not a callable object   # SOT 无法模拟 Tensor.rot90
  ```
- 根因：`paddle.rot90` 的 kernel **不支持 complex64**；且 SOT 符号追踪器无法模拟 `Tensor.rot90` 方法（与 `.repeat()` 同类问题）。**动态图下对复数 rot90 正常**（动态前向已验证通过），仅动转静受限。

### 关键事实
- `full_graph=False`(SOT) **能绕过阻塞 A**（SOT 模拟动态语义，容忍 `self.weights` 重赋值），但**撞阻塞 B**。
- `full_graph=True`(AST) 撞阻塞 A（更早），未触达 B。
- 复数 FFT/einsum/as_complex/setitem **经探针证明 CINN 可编译**，不是问题。
- `get_weight` 含 eval 缓存逻辑（`self.eval_build`，python bool 重赋值，对动转静无害）。

### 候选方向（供调查，**不替所有者拍板**；若某方向需"大量改逻辑/改依赖"请暂停询问）
- B1（最小）：`full_graph=False`(SOT) 下，仅把 3 处复数 `rot90` 替换为数值等价的 `flip + transpose` 组合（flip/transpose 支持 complex 且对动转静友好）。**必须先在动态图对随机复数张量验证 `flip+transpose == paddle.rot90` 数值一致**，否则破坏 G-等变性。仍可能撞后续 to_static 限制（conj / setitem 等），需迭代。
- B2（最大覆盖）：重构 `get_weight` 为返回式（消除前向内 `self.weights` 重赋值）+ 替换 rot90，走 `full_graph=True`。CINN 覆盖最大，但改动最大、等变性风险最高。
- B3（兜底）：放弃模型级动转静，记录阻断原因。

> 调查要点：`rot90(k, axes=[-2,-1])` 的精确 flip/transpose 等价语义要严格推导并数值验证；`k=1/2/3` 各不同。等变性是该模型的核心，**不得改变数值语义**。

---

## 6. 接续约束（再次强调）

- **只解决 §5 的阻塞链**，不重构无关代码、不加未被要求的功能。
- **禁止假想问题**：用最小探针/数值验证驱动，不要凭推测大改。
- **遇疑难大改前暂停询问所有者**（特别是要改 `get_weight` 等变权重核心逻辑或改依赖时）。
- 改动后必须验证：①动态图前向+反向不破坏；②等变性（`eq_check_rt/eq_check_rf`，experiments.py 训练日志会打印 Rotations/Reflections 指标，CINN on/off 应一致）；③CINN 真正编译（看 `Compiling subgraph with CINN backend`）。

---

## 7. 数据集（已确认加载兼容，仅参数差异）

rdb 加载分支（`experiments.py` 搜索 `rdb = `）：靠 `--rdb_super_res`/`--rdb_downsample` 参数化，**无硬编码分辨率**，两数据集本就兼容。

| 数据集 | 路径 | 结构 | 参数 |
|---|---|---|---|
| 小型 SWE | `/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5` | 200 轨 × (101,64,64,1)，333MB | `--rdb_super_res=64 --rdb_downsample=4` → 16×16，`--modes=8` |
| 原版 PDEBench | `/home/lkyu/baidu/PDEBench/pdebench/data_download/data/2D/shallow-water/2D_rdb_NA_NA.h5` | 1000 轨 × (101,128,128,1)，6.6GB | `--rdb_super_res=128 --rdb_downsample=4` → 32×32，`--modes=12` |

两者 rdb 分支均要求 `--T=24`。原版会把全部 1000 轨载入内存（~6.6GB），注意内存；测试可用小 ntrain。

---

## 8. git 未同步（已查清）

- 本地 `main` 领先 `origin/main` **2 个提交未 push**：`7f08436`、`496cb90`（均为 `paddle_random_init/` 相关）。
- 这 4 个文件（`paddle_random_init/{GFNO,utils,generate_weights}.py`、`README.md`）在**工作区被未提交地删除**（`git status` 显示 `D`），但仍在 HEAD。需要时 `git checkout HEAD -- paddle_random_init/` 恢复。
- 本次 CINN 改动尚未提交（所有者未要求提交，**不要擅自 commit/push/动分支**）。

---

## 9. 基准测试方法（所有者指定）

- CINN 有**一次性编译 warmup（约 100s）**，短跑总耗时反而比动态图长。**只看稳态步速/纯训练时间，不看总耗时。**
- 若单 epoch 时间可接受→用单 epoch 对比；否则**积累相同训练步数**消除随机误差，报告：
  - **端到端加速比**（含编译/数据/eval 的 epoch 或累计时间）
  - **纯训练加速比**（仅 forward+backward+step 的稳态时间）
- 报告写入仓库文件（如 `docs/cinn_benchmark.md`）。

---

## 10. 环境与复现命令

- conda env：`paddletorch`；python：`/home/lkyu/miniconda3/envs/paddletorch/bin/python`
- paddle 3.3.0（CUDA 编译），单卡 **RTX 4060 Ti 16GB**
- `libcuda.so cannot open shared object file` 警告**可忽略**（GPU 实际可用，前向在 GPU 跑通）。

动态基线（应正常）：
```
cd G-FNO_paddle && python experiments.py --seed=1 \
  --data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  --results_path=./results/probe_dyn --strategy=teacher_forcing --T=24 \
  --ntrain=64 --nvalid=16 --ntest=16 --model_type=GFNO2d_p4 \
  --modes=8 --width=10 --batch_size=16 --epochs=1 --learning_rate=1e-3 \
  --early_stopping=1 --verbose --rdb_super_res=64 --rdb_downsample=4 --device=gpu
```
CINN 探针（当前在 §5 阻塞 B 处失败）：
```
GFNO_USE_CINN=1 FLAGS_print_ir=false python experiments.py ...（同上，换 results_path）
```

---

## 11. 参考资料

- CINN 官方文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/cinn_cn.html
- **CINN 动转静训练踩坑手册（必读）**：`/home/lkyu/baidu/工作总结/CINN动转静训练踩坑手册.md`
  - 坑一：`.size()/.view()/.repeat()` 兼容层破坏追踪 → 改原生（已处理）
  - 坑二：`Tensor.view()` 反向 `pd_op.view_shape_grad` bug → 用 `.reshape()`（已处理）
  - 坑四：to_static 与 CINN 强绑定 → 单一开关（已采用）
  - 末尾：CINN ~100s warmup、看稳态吞吐的基准教训
- G-FNO 迁移实验记录：`/home/lkyu/baidu/工作总结/G-FNO迁移实验记录.md`
- 仓库内：`G-FNO_paddle/README.md`、`docs/`、`tools/`
