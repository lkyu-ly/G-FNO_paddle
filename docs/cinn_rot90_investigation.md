# GFNO2d CINN 动转静阻塞调查报告

> 调查日期：2026-07-29  
> 范围：仅 `GFNO2d_p4` 的 `GConv2d.get_weight()` 阻塞链；未修改业务代码、依赖或数据链路。

## 结论

当前阻塞不是单一的 `complex64 rot90`，而是按顺序出现的三层 Paddle 3.3.0 兼容问题：

1. `full_graph=True` 下，前向重绑已注册为 Buffer 的 `self.weights` 会触发类型错误。
2. 改用 SOT（`full_graph=False`）后，`paddle.rot90` 的静态 dtype 检查拒绝 `complex64`。
3. 仅替换 `rot90` 后，Hermitian 权重组装中的 `complex64 concat` 会在 CINN 执行时报 `cinn_op.concat` kernel 定义错误。

因此，交接文档中的原始 B1（SOT + 只替换三处 `rot90`）不完整；B2（返回式重构 + `full_graph=True`）即使消除属性重绑，也仍需处理 `complex64 concat`。

推荐采用 **B1+：SOT + `get_weight()` 内局部实/虚分量组装**：

- CINN 开启时改为 `paddle.jit.to_static(model, full_graph=False)`。
- 非 Hermitian 路径只把 `rot90` 改为等价的 `flip + transpose`，群通道循环移位改为 `paddle.roll`。
- Hermitian 路径在 `float32` 实部和虚部上分别完成 `concat`、共轭符号、空间旋转和群通道移位，最后用一次 `paddle.complex(real, imag)` 恢复 `complex64`。
- 保留现有 `self.weights` 缓存/赋值结构，不做返回式重构；修改范围限制在 2D `GConv2d.get_weight()` 及一个私有旋转辅助函数。

该方案的运行时原型已满足：动态图整模型输出逐位一致、SOT 真正触发 CINN、完整 GFNO2d 前向与反向成功、42 个梯度均为有限值。由于它改动等变权重核心逻辑，按交接约束应由所有者确认后再落库。

## 根因证据

### 1. AST 属性重绑

真实 GFNO2d 最小前向在 `full_graph=True` 下稳定失败于：

```text
TypeError: assignment to parameter 'weights' should be of type Parameter or None, but got 'Value'
```

位置是 `GConv2d.get_weight()` 中 `self.weights = paddle.tile(...)`。实例检查显示初始化阶段赋给 `self.weights` 的 Tensor 已被 Paddle 注册进 `_buffers`；AST 构图时右值变为 PIR `Value`，`Layer.__setattr__` 不允许用它重绑该 Buffer。SOT 可保留这一动态状态行为，因此不触发此错误。

### 2. `complex64 rot90`

SOT 在 Hermitian 分支第一处 `rot90(k=2)` 稳定失败：

```text
TypeError: The data type of 'X' in rot90 must be
['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
but received complex64.
TypeError: 'rot90' is not a callable object
```

安装版本源码 `paddle/tensor/manipulation.py:1870` 的 `rot90()` 明确只允许上述实数/整数/bool dtype；同一实现最终本来就是 `flip + transpose`：

- `k=1`：`transpose(flip(x, axes[1]), swapped_axes)`
- `k=2`：`flip(flip(x, axes[0]), axes[1])`
- `k=3`：`flip(transpose(x, swapped_axes), axes[1])`

对随机非方形 `complex64` 张量验证 `k=1/2/3`：替代式与动态图 `Tensor.rot90()` 的前向最大误差、实部梯度最大误差、虚部梯度最大误差均为 `0`；独立 SOT+CINN 前反向也通过。

### 3. `complex64 concat`

把三处 `rot90` 全部替换后，完整 GFNO2d 到达下一层错误：

```text
RuntimeError: (PreconditionNotMet) op [cinn_op.concat]
kernel output args (0) defs should equal op outputs (1)
```

单算子对照只改变 concat dtype：

| 用例 | SOT+CINN 前反向 |
|---|---|
| `float32 concat` | 通过 |
| `complex64 concat` | 同样的 `cinn_op.concat` 错误 |

把拼接结果赋给局部变量或 Layer 属性，结果相同，排除了属性状态副作用。用 complex setitem 替代 concat 也不成立：前向可过，但反向仍生成 complex `cinn_op.concat` 并失败。

## 最小可行方案验证

运行时原型只覆盖主线 `GFNO2d_p4`（`reflection=False`），未写入仓库。其做法是：

1. Hermitian 基础核分别构造实部和虚部：共轭在虚部体现为取负。
2. 所有拼接均保持 `float32`。
3. 90 度旋转分别作用于实部/虚部；输入群通道用 `roll(..., axis=2)` 循环移位。
4. 按原权重维度顺序 `stack + reshape`，裁剪 Hermitian 半谱后再 `paddle.complex(real, imag)`。
5. 非 Hermitian 路径使用经验证的 `flip + transpose` 旋转替代式。

最终证据：

```text
DYNAMIC_MODEL_EQUIVALENCE max_abs_diff=0.0
I... add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
GFNO2D_SOT_FORWARD_BACKWARD_OK output_finite=True gradient_count=42 all_gradients_finite=True
```

探针配置为 `modes=4`、`width=4`、batch 2、16x16 输入；它验证阻塞链和核心数值语义，不替代数据集训练、等变性指标或性能基准。

## 方案比较

| 方案 | 结论 | 原因 |
|---|---|---|
| 原 B1：SOT + 只替换 rot90 | 不足 | 随后必撞 `complex64 cinn_op.concat` |
| B1+：SOT + 局部实/虚组装 | 推荐 | 已通过整模型前反向原型；避开属性重绑、complex rot90、complex concat；改动局限在 2D 权重组装 |
| B2：返回式重构 + AST | 暂不推荐 | 除属性重绑外仍需 B1+ 的 complex 规避；同时要改 `GConv2d`、`GSpectralConv2d` 的权重传递/缓存，风险和改动面更大 |
| SOT 局部断图 | 不可用 | `force_dynamic` 未从 `paddle.jit` 导出；从内部 marker 导入也未阻止嵌套 `get_weight()` 被 SOT 追踪 |
| B3：放弃模型级动转静 | 兜底 | 仅在 B1+ 落库验证失败或无实际加速时采用 |

## 实施风险与验收

主要风险是等变权重维度顺序或共轭符号写错，以及 SOT 子图切分后实际加速不足。实施时应严格限制到 `GFNO2d_p4` 主线，并按以下顺序验收：

1. 同 seed、同参数下，旧动态实现与新动态实现比较输出和参数梯度；目标最大差为 `0` 或机器精度。
2. 运行 `eq_check_rt` / `eq_check_rf`，对比 CINN off/on 指标。
3. 小型 SWE 跑若干 optimizer step，确认 loss、输出、全部梯度有限且日志出现 `Compiling subgraph with CINN backend`。
4. 再跑原版 PDEBench SWE 的相同步数。
5. 最后按交接文档方法测稳态纯训练与端到端加速比；在性能数据出来前不能声称“获得训练加速”。

## 调查命令与工件

- AST/SOT 真实 GFNO2d 失败：`/tmp/probe_gfno2d_static.py`
- rot90 数值与梯度等价、SOT+CINN：`/tmp/probe_complex_rot90_replacement.py`
- concat dtype 对照：`/tmp/probe_cinn_concat_dtype.py`
- 最终整模型运行时原型：`/tmp/probe_gfno2d_b1_runtime_patch.py`
- Paddle 3.3.0 rot90 一手实现：`/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/tensor/manipulation.py:1870`
- CINN 官方说明：<https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/cinn_cn.html>

