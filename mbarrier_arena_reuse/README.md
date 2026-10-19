# BWD mbarrier arena 复用：已确认复现

同卡两个独立 CUDA context 是本轮找到的有效触发条件。旧协议复现 shared
数据损坏；增加 `mbarrier.inval` 后，同条件测试未出现错误。原始未修复 PF
也在 optimizer 之前捕获了 dW1 的 NaN 和约 `-1.04e34` 的有限坏值。

## 最小复现

[minimal.cu](minimal.cu) 不含 GEMM、TMA、cluster、NCCL 或模型：

1. 初始化 barrier，arrive，并等待该 phase 完成。
2. 可选 invalidate，然后将相同 shared 地址覆盖为普通有限数据。
3. 反复读取，检查数据是否改变。覆盖后没有其他 shared writer 或迟到 arrive。
4. 同卡启动第二个独立进程制造 context 竞争；该进程始终使用正确协议，
   与被测进程没有共享地址或 allocation，不能直接写坏其数据。

```bash
# 在空闲 GPU 上运行；脚本自动编译、执行独占对照和三轮双 context 对照。
python run_context.py --device 0 --trials 3
```

已测试 NVIDIA L20X、SM90a/132 SM、1620 MHz，driver 570.148.08、
CUDA 12.9.86。最小程序只需 CUDA；当前布局每 CTA 使用 213120 B shared，
32 threads，保留原 9 个 barrier 的地址，每 SM 一个 CTA。

[原始结果](results_context_minimal_20261019.jsonl)，每个进程 300 次 kernel、
每次最多保持 10 ms：

| 场景 | 未 invalidate | invalidate |
|---|---:|---:|
| 独占 GPU | 0 个坏槽位 | — |
| 双 context，重复 3 轮 | 每轮 1188/1188 个槽位损坏 | 每轮 0 个坏槽位 |

槽位数是 CTA × barrier 的累计去重统计，不是错误发生次数。另已验证
单 CTA、单 barrier、8 B shared 的缩小变体也能复现；它不是上述结果文件的配置。

普通数据 `0x3c003b803b003a80` 被改成 `0xfffff800001ffffe`。
后者按小端 BF16 解读，包含 NaN 和 `-1.0384593717069655e34`。
**观测确认了未 invalidate 的 arena 复用会损坏数据；context 切换期间具体
如何保存/恢复 barrier 内部状态，仍是机制推断，不声称已追踪硬件内部过程。**

## 对齐原 PF 的流水线消融

[repro.cu](repro.cu) 保留双 math warpgroup、3-stage TMA/WGMMA、cluster2、
多 dW tile 和 STSM/TMA 写回，分别控制 producer-tail drain 与旧 barrier
invalidate。没有人为添加旧 barrier 的迟到写入。该程序用于正确性诊断，
不是性能 benchmark；其编译有 WGMMA 串行化提示。

```bash
make CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/opt/cutlass
# cluster ctas steps repeats drain invalidate device [gemm] [tiles]
# 两个命令需在同一空闲 GPU 上并发执行，分别使用独立进程。
build/repro 2 0 256 100 0 0 0 1 256 &
build/repro 2 0 256 200 1 1 0 1 256 &
wait
```

同卡 context 竞争下，每项被测配置 100 次 kernel：
[原始结果](results_context_stream_20261019.jsonl)。

| drain | invalidate | 坏 BF16 元素 | 其中非有限 |
|---:|---:|---:|---:|
| 0 | 0 | 3860 | 1930 |
| 1 | 0 | 2364 | 1182 |
| 0 | 1 | 0 | 0 |
| 1 | 1 | 0 | 0 |

所有坏元素均落在旧 barrier 区域；GEMM accumulator 检查未报错。
本例中 invalidate 是决定性修复，单加 drain 不够；仍保留生产代码的 drain，
因为它负责结束未完成事务，与 invalidate 的生命周期职责不同。

## 原生 PF 验证

[native_stress.py](native_stress.py) 使用原始 native extension，EP4、Qwen
H2048/I512/E256/top8，保留 16 层 FWD cache 后逆序 BWD，tokens 在
8191/20003/32767/17 间变化。不含 optimizer、AReaL 或权重更新。

- 未修复版在第 218 轮、rank 2、layer 11 的 dW1 捕获 48 个坏值：
  36 个 NaN、12 个约 `-1.04e34` 的有限值。local expert 1，
  行 608/736、列 1792..1827，均对应 tile 内第 96 行的旧 barrier 区域。
  [原始日志](native_old_ep4_20261019.log)。
- 修复版完成 500 轮、**每 rank 8000 次 BWD**，未检测到坏值；
  各 GPU 另启动的正确协议 context 竞争进程也未检测到坏值。
  [原始日志](native_fixed_ep4_20261019.log)。
- 原生旧版此次失败发生在共卡压力实验期间，但不据此断言坏值所在 GPU
  的具体抢占来源；受控双 context 因果对照由上述独立程序完成。

旧 extension：WeftMoE 的
`build/pf_main_timeline/qwen_areal_pf_dt_20261016/_runtime.cpython-312-x86_64-linux-gnu.so`，
SHA256 `d5aee6d827c4c9011b7d8ee9be300e848cd829054fddbd98646ea305f6b81eaf`。
修复版：`build/pf_main_timeline/qwen_areal_barrier_fix_20261018/` 下同名 extension。
运行需要通过 `WEFTMOE_TENSOR_EXTENSION` 选择二进制，并用
`torchrun --standalone --nproc-per-node=4 native_stress.py --rounds 500 --layers 16`。
脚本检查非有限和超大有限值，不做模型完整数值精度基准。

## 与原 RL 故障的对应关系

原 Qwen3.5-35B-A3B 长跑第 7 次权重更新后出现非有限 logprob；
actor CPU 权重与 Adam 状态已损坏。rank 3、layer 13、expert 99 的
FC1 BF16/master/m1 各 72 个 NaN，m2 有 96 个非有限值；另 24 个位置
m1 约 `-5.19e32`。坏值集中在行 608/736/864/992、列 1280..1315。

| shared 对象 | 字节偏移 |
|---|---:|
| 3-stage A/B operand | 0..147455 |
| 原 pull scratch | 147456..159743 |
| 原 9 个 barrier | 159744..159815 |
| 新 dW 输出 tile | 147456..212991 |
| 新 dW barrier | 从 212992 开始 |

输出按 N64 atom 排列，旧 barrier 恰好覆盖 M128 tile 内第 96 行前 36 个 BF16。
独立程序、原生 PF 与 RL 现场的损坏位置和坏值类型一致。修复提交是
WeftMoE `9b38230`，本轮不修改生产 kernel。

[PTX 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-inval)
要求存储改作其他用途前 invalidate；普通 shared store 不能替代它。

## 早期未复现记录

提交 `3dabcff` 的独占 GPU 版本曾运行 28 万次而未捕获错误：
[全 SM](results_l20x_20261019.jsonl)、
[stage 边界](results_stage_edges_l20x_20261019.jsonl)、
[单 CTA](results_single_cta_l20x_20261019.jsonl)、
[sanitizer](sanitizer_l20x_20261019.log)。
这些是旧版本的历史负结果，缺少本轮有效的多 context 触发条件，
不能证明旧协议安全，也不再代表当前复现状态。

本目录代码为 Apache-2.0；流水线版复用相邻目录的 `hopper.cuh`，见
[LICENSE](../tma_store_vs_vector/LICENSE) 和
[THIRD_PARTY_NOTICES](../tma_store_vs_vector/THIRD_PARTY_NOTICES.md)。
