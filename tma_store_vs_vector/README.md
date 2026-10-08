# TMA store vs vector：吞吐与 ring stage 复用

## 先看结论

1. **独立连续 store：基本持平。** 固定写出 512 MiB、8 KiB/tile，132–8448
   CTA 下差距约在 2% 以内，不能说 vector 或 TMA 天然更快。
2. **独立跨行 store：高并发时 TMA 占优。** 行跨度 8192 B、528 CTA 时，
   TMA 耗时低约 12%；132 CTA 时仍基本持平。这不是多写数据或 sector 利用率低。
3. **复用 ring stage：vector 更快。** 此样例的 copy 工作流快约 9%，加入
   WGMMA 后快约 1–2%；等价地，TMA 比 vector 多耗时约 10% / 1–2%。
4. **启示：同时看搬运吞吐和源 buffer 的生命周期。** TMA 能异步搬运，
   不代表 ring stage 可以立即覆盖；增加在途请求也可能减少 producer 的超前空间。

这是单 GPU CUDA 最小诊断，不是完整 MoE 性能测试。没有 WeftMoE runtime、
Python、NCCL 或跨机依赖。两种场景不能直接相减，不能用其中一个推断另一个。

## 1. 独立 store：同样的字节数，不同的目的地址

[store_repro.cu](store_repro.cu)：shared tile 一次初始化后保持不变，没有
GM→SM refill、ring、WGMMA 或 stage credit。Vector 每次实际读取 shared；
TMA serial 每 tile commit + full wait，queued 最多八组在途，退出前 full wait。

8 KiB tile 是 64 行 × 64 个 BF16。一个 warp 每线程写 16 B，总计 512 B：

| 线程 | 连续目的地址偏移（B） | 跨行目的地址偏移（B） |
|---|---|---|
| 0–7 | 0–127 | 0–127 |
| 8–15 | 128–255 | 8192–8319 |
| 16–23 | 256–383 | 16384–16511 |
| 24–31 | 384–511 | 24576–24703 |

两者都是 16 个写满的 32 B sector。跨行来自宽度 4096 BF16 的矩阵中，
每行只写当前 K64 小片；**不是 shared bank conflict，也不是内存分配碎片**。
TMA/vector 写相同地址，TMA 不会把跨行目标变连续。

固定写出 **512 MiB**，256 threads/CTA，以下单位 **μs**：

| 布局 | launch CTA 数 | vector | TMA serial | TMA queued |
|---|---:|---:|---:|---:|
| 连续 | 132 | 131.208 | 132.184 | 130.768 |
| 连续 | 264 | 131.288 | 131.432 | 131.112 |
| 连续 | 528 | 134.456 | 134.192 | 133.496 |
| 连续 | 1056 | 133.176 | 131.376 | 131.432 |
| 连续 | 2112 | 131.864 | 129.744 | 130.944 |
| 连续 | 4224 | 128.280 | 128.112 | 128.456 |
| 连续 | 8448 | 125.728 | 124.792 | 125.112 |
| 跨行 | 132 | 134.888 | 135.192 | 134.504 |
| 跨行 | 528 | 152.344 | 134.032 | 136.640 |

CTA 数增加不增加总数据量，也不等于同比增加同时驻留数。两种模式都计到
kernel 完成，包含 shared 初始化；分配/校验在计时外，没有扣掉 `none` 时间。

**归因边界：** 地址分布和请求处理方式会影响排队/吞吐，但尚未锁定跨行
差距的具体硬件瓶颈。旧版计数器中，连续/跨行高并发的 LG throttle 都约
28%，不能用这个指标单独证明差距；也不能把“跨行”直接等同于不合并访存。

**已修正的坑：** 早期仅写 C++ `asm volatile`，ptxas 仍将不可变源的 LDS
外提到循环外。当前用 PTX `ld.volatile.shared`，反汇编确认 LDS 在循环内；
NCU 的 shared-load wavefront 数也对应完整 512 MiB 读取。上表全部来自
修正版本，不包含旧版的 register→global 数据。

## 2. Ring stage 复用：更快的 store 不一定带来更快的流水线

[ring_stage_repro.cu](ring_stage_repro.cu) 保留 3-stage、cluster2 A multicast、
M128/N256/K64、H4096、384 threads、226304 B dynamic shared。`copy`
保留 GM→SM refill 与 ring 同步；`gemm` 再加入真实 BF16 WGMMA，保存 X
位于 WGMMA commit 与 wait 之间。每个 M-block 只有轮转选中的一对 CTA
保存 X，各保存 M64；不包含 route planner、FC2、return/reduce。

同步关系：producer 等 `empty` 才覆盖 stage，TMA load 完成发布 `full`，
consumer 等 `full` 后计算/保存，再归还 `empty`；阶段回绕时翻转 phase。
cluster 的两个 CTA 都消费结束，multicast 源 stage 才可复用。

| mode | 何时允许复用源 stage | 何时保证目标写完 |
|---|---|---|
| `vector` | 所有消费者完成 LDS，且 WGMMA 读完 | kernel 完成后读取结果 |
| `tma_full` | 每 K full wait + warp join，再等 WGMMA | 每 K 完整等待 |
| `tma_read` | 每 K read wait + warp join，再等 WGMMA | M-task 末尾 full wait |
| `tma_pipelined` | 下一 K 等前一组源读完，再归还前一 stage | M-task 末尾 full wait |

Vector 的 LDS 完成后，待写数据已进入寄存器，后续 GM 写出不再引用 shared。
TMA 则必须确认异步源读取完成。把目标写出完成延后，不能取消源读取等待。

以下为 lane0/M32/fence-on，同轮各模式使用同一 kernel binary，单位 **ms**：

| 工作流 | routed rows | vector | TMA full | TMA read/full 拆分 | TMA pipelined |
|---|---:|---:|---:|---:|---:|
| ring copy | 49152 | 0.245656 | 0.269320 | 0.269896 | 0.350792 |
| ring copy | 122880 | 0.602504 | 0.661352 | 0.662104 | 0.859272 |
| FC1 GEMM | 49152 | 2.510408 | 2.544712 | 2.547568 | 2.651360 |
| FC1 GEMM | 122880 | 6.152200 | 6.281896 | 6.289552 | 6.491216 |

49152/122880 是每卡 routed rows，对应均衡 EP8、top-k6 下 nominal
8K/20K source tokens/rank 的计算量。没有随机路由，也不是完整 FWD。

消融与启示：

- **拆分 read/full wait 没有稳定收益。** 不能据此认定目标写出等待是主瓶颈。
- **延后一个 K 归还 stage 反而更慢。** 更多 store 在途，也意味着 producer
  更少可用 stage；不能只看 store 的 overlap。
- **elect、两次 M32 合并 M64、删除此路径多余的 source fence，均未逆转结论。**
  Fence 消融仅适用于这里已等待完成的 TMA-load 源，不能推广到普通 shared store。
- GEMM 的 vector/TMA full：Tensor active **83.97%/82.46%**，GMMA arrive
  wait **0.77%/1.24%**；DRAM 写出相同，读取量却变化。证据支持 stage 生命周期、
  warp 推进与缓存复用相互影响，不支持把全部差距分配给某一条 wait。
- 不能说“TMA 单次延迟一定大于 WGMMA”。这里测的是完整流水线，不是逐指令延迟。

## 3. 编译与复现

已验证环境：NVIDIA L20X（SM90a、132 SM），CUDA **12.9.86**，CUTLASS
headers **4.2.0**。Ring 需要 cluster2 和至少 226304 B dynamic shared/CTA。

```bash
cd tma_store_vs_vector
make -j2 CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/opt/cutlass
make smoke                 # 小规模正确性检查，GPU0，不作为性能数据

# 全部表格与消融：GPU0、每模式40个样本，配置正反各一轮
bash run.sh 0 40           # 新日志放入 build/results.XXXXXX，不覆盖历史数据

# 单独运行：rows width output_MiB grid_multiplier samples device [mode]
build/store_repro 64 64 512 8 40 0
build/store_repro 64 4096 512 4 40 0

# copy|gemm routed_rows lane0|elect tma_rows fence samples device [mode]
build/ring_stage_repro copy 49152 lane0 32 1 40 0
build/ring_stage_repro gemm 122880 lane0 32 1 40 0
build/ring_stage_repro gemm 49152 elect 64 0 40 0
```

`store_repro` 支持 4/8/32 KiB tile（rows32/64/256），本页报告修正后的
8 KiB 实验。可选 mode 见前面的表；ring 另有 `none`（不保存 X），store
另有 `none`（只做 shared 准备）。`tma_pipelined` 仅支持稳定的 lane0 issuer。
程序不会自动改时钟；测试前确认设备空闲，按本机管理方式设置 SM 1620 MHz。

## 4. 测量口径与可追溯性

- 2026-10-08，GPU0，SM **设置** 1620 MHz、memory 3201 MHz；每模式
  8 warmup + 40 samples、mode 交替顺序、计时外间隔 2 ms。表中为两轮
  median 的均值。分配/初始化输入/数值校验均在计时外。
- `results/ring_r{0,1}.jsonl` 与 `results/store_r{0,1}.jsonl` 保留所有样本；
  只去掉原日志的 `SAVED_X_REPRO` / `STORE_REPRO` 前缀，未修改数值。
  Ring 数据来自 WeftMoE `7a0950d` 实验，store 来自修正版本 `8606637`。
- Ring 检查 saved X bitwise、GEMM bitwise 对照，并用 CPU 独立计算 32 个
  dot products；store 逐 word 检查。`none` 不检查未写出的 saved/output。
  原样例定向 memcheck/racecheck 通过，不代表完整 MoE 或所有参数均已验证。
- **锁频设置不等于 kernel 内严格恒频。** NVML 采样是 1620 MHz；独立 NCU
  的 ring GEMM 有效频率约 1440–1454 MHz，store/grid1056 为 1548.5/1581.1 MHz。
  Profiler 时间不混入表格；1–2% 级差异谨慎解读，不按设置频率倒推峰值 FLOPS。
- 提取时未重新验收全套性能：本目录已独立编译、通过 `make smoke`；两个样例
  的 SASS 指令序列分别与历史测量产物一致。GEMM 仍 REG168/local56，不能将
  这个局部诊断的绝对时延当成正式 PF 的时延。

同步语义参考 [CUDA async copies](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html)，
计数器口径参考 [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#quantities)。
来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
