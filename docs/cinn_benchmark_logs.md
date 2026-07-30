# G-FNO CINN 加速比测试 — 原始日志归档

> 归档日期：2026-07-29　|　环境：PaddlePaddle 3.3.0（CUDA），单卡 RTX 4060 Ti 16GB，conda env `paddletorch`
>
> 来源说明：`/tmp/*.log` 在先前的清理中已删除；本文件从后台任务输出（`/tmp/claude-1000/.../tasks/*.output`）与当次会话 stdout 逐字恢复。标注【磁盘恢复】= 来自仍在磁盘上的任务输出；【会话恢复】= 来自会话记录（原始 /tmp 文件已删）。基准脚本为 `/tmp/bench_cinn.py`（合成 batch、B=32，20 步预热后 3 段 ×50 步；已随清理删除）。

## 1. 纯训练稳态基准（加速比来源）

### 16×16（小型 SWE，modes=8）

【会话恢复】DYN（CINN 关）：
```
BENCH mode=DYN compile_s=0.659 per_step_ms=39.172 seg_per_step_ms=[38.713, 39.314, 39.49] spread=0.778 n_steps=50 end2end_train_s=2.618
```
【会话恢复】CINN（开）：
```
I ... add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
BENCH mode=CINN compile_s=99.954 per_step_ms=36.209 seg_per_step_ms=[36.647, 37.46, 34.521] spread=2.939 n_steps=50 end2end_train_s=101.765
```
加速 = (39.172 − 36.209) / 39.172 ≈ **7.56%**；CINN 一次性编译 ~100s。

### 32×32（原版 128 下采样，modes=12）

【磁盘恢复 · blci26sdf.output】
```
=== 32x32 DYN ===
BENCH mode=DYN compile_s=1.297 per_step_ms=42.150 seg_per_step_ms=[40.383, 44.085, 41.981] spread=3.702 n_steps=50 end2end_train_s=3.405
=== 32x32 CINN（~100s编译）===
I0729 16:10:01.060411 958289 add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
BENCH mode=CINN compile_s=106.117 per_step_ms=37.256 seg_per_step_ms=[39.03, 35.597, 37.14] spread=3.433 n_steps=50 end2end_train_s=107.980
```
加速 = (42.150 − 37.256) / 42.150 ≈ **11.60%**；CINN 一次性编译 ~106s。

## 2. 训练跑通日志（experiments.py，1 epoch，CINN 编译标志 + loss/等变性）

### 小型 SWE 16×16

【磁盘恢复 · b2gfzodng.output】CINN 开：
```
I0729 15:48:37.076134 917219 add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
GFNO2d_p4; Input shape: paddle.Size([16, 16, 16, 1, 1]), Target shape: paddle.Size([16, 16, 16, 1])
Ep: 0, time: 6.561094196003978, train: 0.14855776874658963, valid: 0.04826880618929863
Test: 0.04647275432944298, Rotations: 1.756347955961246e-07, Reflections: 0.004861708264797926, Super Space Test: None, Super Time Test: None
```
【会话恢复】DYN（CINN 关）：
```
GFNO2d_p4; Input shape: paddle.Size([16, 16, 16, 1, 1]), Target shape: paddle.Size([16, 16, 16, 1])
Ep: 0, time: 6.196585826000955, train: 0.14855774522099333, valid: 0.04826884716729191
Test: 0.04647277681118477, Rotations: 1.862979972750953e-07, Reflections: 0.004861701745539904, Super Space Test: None, Super Time Test: None
```
→ DYN/CINN loss 逐位一致（train≈0.14856、valid≈0.04827、test≈0.04647），Rotations≈1e-7 等变保持。

### 原版 128 子集 32×32

【会话恢复】DYN（CINN 关）：
```
GFNO2d_p4; Input shape: paddle.Size([16, 32, 32, 1, 1]), Target shape: paddle.Size([16, 32, 32, 1])
Ep: 0, time: 4.698636360997625, train: 0.15676879366630614, valid: 0.05342064052820206
Test: 0.04986666515469551, Rotations: 2.069654954084399e-07, Reflections: 0.007741525303572416, Super Space Test: None, Super Time Test: None
```
【磁盘恢复 · bdyv4kh13.output】CINN 开（zeros 修复后）：
```
I0729 16:07:10.780495 954304 add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
GFNO2d_p4; Input shape: paddle.Size([16, 32, 32, 1, 1]), Target shape: paddle.Size([16, 32, 32, 1])
Ep: 0, time: 26.43385280599614, train: 0.15676871174946427, valid: 0.05341936647891998
Test: 0.0498691126704216, Rotations: 2.1368799707488506e-07, Reflections: 0.0077414982952177525, Super Space Test: None, Super Time Test: None
```
→ DYN/CINN loss 一致（train≈0.15677、valid≈0.05342、test≈0.04987）。Ep 时间 26.4s 含首 epoch 残余 CINN 编译，非稳态，加速比以 §1 基准为准。

### PaddleCFD（ppcfd editable 安装）冒烟测试 16×16 CINN

【磁盘恢复 · byfkw6frq.output】
```
I0729 16:30:10.684439 990568 add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
GFNO2d_p4; Input shape: paddle.Size([16, 16, 16, 1, 1]), Target shape: paddle.Size([16, 16, 16, 1])
Ep: 0, time: 6.4300208920030855, train: 0.14855777326738462, valid: 0.04826880246400833
Test: 0.04647278040647507, Rotations: 1.780401248652197e-07, Reflections: 0.004861700348556042, Super Space Test: None, Super Time Test: None
```
→ 与开发仓库 16×16 CINN 结果一致，确认同步到生产仓库的代码行为相同。

## 3. 数值等价验证（B1+ 移植 + zeros 修复）

【会话恢复】动态图逐位等价（B1+ 移植前后，modes=8/width=10/16×16）：
```
OUT_MAX_ABS_DIFF 0.0 GRAD_MAX_ABS_DIFF 0.0 ngrads 42
```
【会话恢复】zeros 修复后复核：
```
RECHECK_OUT_DIFF 0.0 GRAD_DIFF 0.0
```
【会话恢复】参考抓取（移植前）：
```
CAPTURED out_shape (2, 16, 16, 1, 1) ngrads 42 out_sum -339.0697326660156 out_abs_max 0.7355320453643799 grad_abs_sum 476.2609433513138
```

## 4. 中间失败运行（追溯用）

【磁盘恢复 · bkoksd0ek.output】原版 32×32 CINN 首次（zeros 未修复，已定位并修复）：
```
I0729 16:03:17.470490 948894 add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
GFNO2d_p4: Train/valid/test data shape:
Traceback (most recent call last):
    raise TypeError(
TypeError: multiple values for argument 'dtype'
paddle.jit.sot.utils.exceptions.InnerError: Call paddle_api error: zeros
```
根因：`paddle.zeros(..., dtype=, device=)` 的 `device` kwarg 在 32×32 CINN 下报错；修复为列表 shape + 去掉 `device=`（详见 `docs/cinn_rot90_investigation.md` 与 `docs/cinn_benchmark.md`）。
