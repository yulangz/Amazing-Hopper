# BWD mbarrier arena 复用：独立诊断

**当前状态：保留了原问题的共享内存布局和关键交接流程，但尚未独立触发原来的 NaN。**
这是用于继续缩小问题的可运行最小诊断，不能称为已经成功复现故障。
关闭修复也通过，既不能证明旧协议正确，也不能单凭本实验确认现场 NaN 的根因。

## 现场现象

WeftMoE + AReaL 的 Qwen3.5-35B-A3B 长跑中，第 7 次权重更新后 rollout
出现非有限 logprob。检查发现，训练 actor 的 CPU 权重和 Adam 状态已经损坏，
不是仅发生在权重发布或推理端。

- rank 3、layer 13、expert 99 的 FC1：BF16/master/m1 各 72 个 NaN；
  m2 有 96 个非有限值，另 24 个位置的 m1 约为 `-5.19e32`。
- 坏值集中在行 `608/736/864/992`、列 `1280..1315`：均为 M128 tile
  内第 **96 行、连续 36 个 BF16**，恰好对应旧 9 个 mbarrier 的位置。

| shared-memory 对象 | 字节偏移 | 关系 |
|---|---:|---|
| 3-stage A/B operand | 0..147455 | M128/N256/K64，BF16 |
| 原 pull scratch | 147456..159743 | 3 × 4096 B |
| 原 full/empty/pull_full | 159744..159815 | 9 × 8 B |
| 新 dW 输出 tile | 147456..212991 | 128 × 256 × 2 B，覆盖以上 scratch/barrier |
| 新 dW barrier | 从 212992 开始 | 不与输出重叠 |

输出按 N64 atom 排列、128 B swizzle。旧 barrier 相对输出起点为 12288 B，
即 atom 0 的第 96 行；该行的 swizzle 不改变这段位置。现场 FC1 的全局列
从 1280 开始，对应这个 tile 的前 36 列。

旧实现没有在转作普通数据前 invalidate 原 barrier，这是明确的协议错误。
[PTX mbarrier.inval 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-inval)
要求先结束相关操作并使 barrier 失效，再将其存储用于其他用途；普通 shared store
不能替代 invalidate。**“硬件晚到写回覆盖 dW”仍是待验证的解释，不是已证实的机制。**

WeftMoE 修复 `9b38230` 同时增加 producer-tail drain 和 9 个旧 barrier 的
invalidate；本程序把两者拆成独立开关。原始排查记录见 WeftMoE 的
`docs/areal_qwen35.md`，没有把模型权重或训练数据复制进这个目录。

## 最小程序保留了什么

[repro.cu](repro.cu) 只依赖 CUDA/CUTLASS，不加载模型、Python、AReaL、
NCCL、RDMA 或 optimizer。

1. 384 threads/CTA，两个 math warpgroup、一个 TMA producer、三个 pull warp；
   3-stage TMA load，cluster2 时保留 A multicast 和远端 empty arrivals。
2. 激活阶段结束后 grid join；可选 tail drain、可选旧 barrier invalidate；
   初始化独立的 12 个 dW barrier，再运行一轮 operand pipeline。
3. dW math 完成后直接 STSM.x2 写到复用 arena，再 fence、math-only barrier、
   TMA store。写回前没有额外的 dW 全网格/全 CTA join；最后才 join 退出。
4. 输出使用随行、列、CTA、迭代变化的有限 BF16 位模式，逐元素比较完整 tile，
   另外统计非有限值和原 barrier 区域坏值。`gemm=1` 加入真实 WGMMA，并检查
   最后阶段每线程的一个 accumulator；输出位模式仍独立于 GEMM 数值。

`drain` 只切换激活阶段的 producer-tail 等待，dW 尾部始终 drain。
`invalidate` 只切换旧 9 个 barrier 的失效指令；不故意发出迟到 arrive 或制造
另一种写后竞态来冒充原故障。软件 grid barrier 要求全部 CTA 可同时驻留，
因此限制为最多每 SM 一个 CTA（本例 shared 为 213120 B）。

简化掉了真实路由/不等长任务、多个 dW 输出 tile、NVLink return、Megatron
及 offload 时序。CUDA 12.9 的 GEMM 变体另有 WGMMA 函数边界串行化和
setmaxnreg 提示；GEMM 变体 local-memory 为 0 B，copy 变体为 288 B/线程。
不能认为流水线时序与完整 PF 相同，本程序不用于比较性能。

## 运行

已测试：NVIDIA L20X，SM90a/132 SM，1620 MHz；driver 570.148.08，
CUDA 12.9.86，CUTLASS 位于 `/opt/cutlass`。需要支持本例 shared 容量的 SM90a GPU。

```bash
make CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/opt/cutlass
# 参数：cluster ctas steps repeats drain invalidate device [gemm]
# ctas=0 自动使用全部 SM；先在一张空闲 GPU 上运行。
build/repro 1 1 32 10000 0 0 0 1   # 单 CTA，关闭两项修复
build/repro 2 0 32 10000 0 0 0 1   # cluster2，全 SM，旧协议
build/repro 2 0 32 10000 1 1 0 1   # 同布局，开启两项修复
bash run.sh 0 10000 32              # copy/GEMM × cluster1/2 × drain/invalidate
```

每次打印一行 JSON。`mismatches/nonfinite/overlap/gemm_mismatches` 为累计错误数，
全零表示没有捕获坏值；退出码 0=未捕获，2=数值错误，1=配置/CUDA 错误。
只有数值错误才算捕获，不把 timeout 或编译失败算作复现。

## 本轮结果（2026-10-19）

- [16 组全 SM 消融](results_l20x_20261019.jsonl)：32 steps，每组 10000 次。
- [8 组 stage 边界](results_stage_edges_l20x_20261019.jsonl)：cluster2/GEMM，
  3/31/33/128 steps，两项修复同开或同关，每组 10000 次。
- [4 组单 CTA 消融](results_single_cta_l20x_20261019.jsonl)：GEMM，32 steps，
  drain/invalidate 四种组合，每组 10000 次。
- [racecheck/synccheck](sanitizer_l20x_20261019.log)：双 CTA、32 steps、2 次，
  旧协议与完整修复分别运行。

上述数值测试未捕获坏值，sanitizer 未报告错误。**共 28 万次 kernel 调用的
负结果，不是原问题已复现或已被排除的证据。** 后续若继续缩小现场故障，应从
原生 PF 的真实任务流增加缺失条件，并捕获首次 dW 损坏；目前不能把 drain 与
invalidate 的现场贡献分别量化。

本目录代码为 Apache-2.0；复用相邻目录的 `hopper.cuh`，见其
[LICENSE](../tma_store_vs_vector/LICENSE) 和
[THIRD_PARTY_NOTICES](../tma_store_vs_vector/THIRD_PARTY_NOTICES.md)。
