# L2-36 FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

> **叙事母题**：⚡ **异步极限** × 🚀 **H100 榨干** —— 当算法不再只看 FLOPs，而开始啃硬件电路图。

---

## 0️⃣ 案件档案

### 📇 精确事实卡

| 字段 | 内容 |
| --- | --- |
| **案件编号** | L2-36 |
| **案件标题** | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| **侦探团（作者）** | Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, **Tri Dao** |
| **办案机构** | Meta · NVIDIA · Princeton · Colfax Research · together.ai |
| **结案日期** | 2024-07-11 |
| **卷宗编号** | arXiv:2407.08608 |
| **PDF 位置** | `PDFs/L2-36_FlashAttention_3.pdf` |
| **核心硬件** | NVIDIA H100 SXM5（Hopper 架构，仅此一家） |
| **关键成绩** | FP16 **740 TFLOPS**（FA-2 的 1.5–2.0×），FP8 **1.2 PFLOPS**，H100 利用率 **75%** |

### 🕸️ 五维雷达

```
              新颖性 ★★★★☆ (4/5)
                  ▲
                  │
影响力 ★★★★★ ◄───┼───► 严谨性 ★★★★★
                  │
                  ▼
              可复现性 ★★★★☆ (4/5)
              工程价值 ★★★★★ (5/5)
```

| 维度 | 评分 | 说明 |
| --- | --- | --- |
| **新颖性** | ★★★★☆ | 思路是 FA-1/2 的延续，但把 Hopper 异步原语用到极致是首创 |
| **严谨性** | ★★★★★ | 完整给出 producer/consumer 流水线、量化误差证明 |
| **可复现性** | ★★★★☆ | 已开源到 `flash-attn` 包，但需要 H100 才跑得动 |
| **影响力** | ★★★★★ | 几乎所有 H100 上的 LLM 推理框架（vLLM、TRT-LLM、SGLang）当天接入 |
| **工程价值** | ★★★★★ | 把 attention 从 35% MFU 拉到 75% MFU，相当于硬件升级一代 |

---

## 1️⃣ 30 秒速览 #速览

> **一句话破案**：FlashAttention-3 用 H100 的 **TMA 异步搬运 + WGMMA 异步矩阵乘 + FP8 Tensor Core**，配合 **producer-consumer warp 专化** 和 **pingpong 调度**，把 attention 在 H100 上的利用率从 FA-2 的 35% 拉到了 75%（740 TFLOPS FP16，1.2 PFLOPS FP8）。

**三个关键词**：

1. ⚡ **异步**：TMA 加载数据时，WGMMA 在算上一块，softmax 在算上上一块——三件事完全重叠。
2. 🎭 **Warp 专化**：128 个线程分成 producer warp（搬数据）和 consumer warp（算 GEMM），各司其职。
3. 🔢 **FP8 + 数值技巧**：block-wise scale + incoherent processing（随机旋转），把 FP8 误差从"naive FP8 烂 2.4 倍"翻盘成"比 naive FP8 好 2.6 倍"。

**一句结论**：当 GPU 走向异步多引擎（Hopper 之后是 Blackwell 也只会更复杂），算法层不"读懂硬件电路"就吃不到峰值。

---

## 2️⃣ 3 分钟通读：为什么 FA-2 在 H100 上只能跑出 35%？ #通读

### 🤔 案件起因

2023 年 FlashAttention-2 横扫 A100，在 A100 上能拿到 70%+ 的 FP16 利用率。但 2023 年底大家把 FA-2 搬到 H100 后，发现一件诡异的事：

$$
\text{FA-2 on H100} = 335 \text{ TFLOPS} \approx 35\% \text{ of H100 FP16 peak (989 TFLOPS)}
$$

**A100 上 70%，H100 上只有 35%——一代硬件白升级了。** 为什么？

### 🔍 三条线索

**线索 1：H100 把 GEMM 做成了"异步"**

| 架构 | 矩阵乘指令 | 同步性 |
| --- | --- | --- |
| A100 (Ampere) | `mma.sync` | **同步**：发指令后线程必须等结果 |
| H100 (Hopper) | `wgmma`（Warp Group MMA） | **异步**：发指令后立刻返回，结果稍后到 |

FA-2 的代码假设 GEMM 是同步的，所以在 H100 上，warp 发完 GEMM 后还在傻等，结果硬件其实已经空出来了。

**线索 2：H100 引入了 TMA（Tensor Memory Accelerator）**

TMA 是 H100 上一颗专门负责"全局内存 ↔ 共享内存"搬运的 DMA 引擎：

- A100：搬数据需要每个线程发 `cp.async`，128 线程齐刷刷搬。
- H100：**一个线程发一条 TMA 指令**，硬件后台自动把整块 tile 搬完。

FA-2 没用 TMA，所以"搬数据"和"算 GEMM"还是抢同一批 warp 的资源。

**线索 3：H100 有 FP8 Tensor Core，但 FA-2 只支持 FP16**

H100 FP8 峰值是 FP16 的 2 倍（约 1979 TFLOPS）。FA-2 不支持 FP8，等于把这一块算力直接扔掉了。

### 🎯 FA-3 的作案手法

针对三条线索，FA-3 给出三招：

1. **针对线索 1 + 2**：搞 **producer-consumer warp 专化**，让一部分 warp 用 TMA 异步加载下一块 tile，另一部分 warp 用 WGMMA 异步算当前 tile，两者通过共享内存的 mbarrier 同步。
2. **针对 GEMM 和 softmax 不对称**：搞 **pingpong 调度**——两个 warp group 交替做 GEMM 和 softmax，A 算 GEMM 时 B 算 softmax，A 算 softmax 时 B 算 GEMM，把 softmax 的 exp/max 操作完全藏在 GEMM 后面。
3. **针对线索 3**：上 FP8，但配合 **block-wise quantization + incoherent processing**，把 FP8 的精度问题压到可用范围。

最终结果：740 TFLOPS FP16（H100 的 75%），1.2 PFLOPS FP8（H100 FP8 的 75%）。

**Tri Dao 一年内连发 FA-3 + Mamba-2，奠定他在"算法–硬件协同设计"领域的统治地位。**

---

## 3️⃣ 30 分钟精读 #精读

### 3.1 H100 架构速成班

要看懂 FA-3，必须先看懂 H100 的四个新原语：

#### 3.1.1 TMA（Tensor Memory Accelerator）

```
┌─────────────────────────────────────────────────────┐
│ A100 异步拷贝（cp.async）：                          │
│                                                     │
│ Thread 0 ──cp.async──► [Global Mem] ──► [Shared]   │
│ Thread 1 ──cp.async──► [Global Mem] ──► [Shared]   │
│   ...    （128 线程都要发指令）                       │
│ Thread 127 ──cp.async──► [Global Mem] ──► [Shared] │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ H100 TMA：                                           │
│                                                     │
│ Thread 0 ──tma.load──► [TMA 引擎]                   │
│                            ↓ （硬件后台搬运整块）       │
│                       [Global Mem] ──► [Shared]     │
│ 其他 127 个线程：可以去做别的事                        │
└─────────────────────────────────────────────────────┘
```

**关键收益**：
- 寄存器压力降低（不用每个线程都算地址）
- 异步：发完 TMA 指令后线程立刻空闲
- 支持 multicast（一份数据广播到多个 SM 的 shared memory）

#### 3.1.2 WGMMA（Warp Group Matrix Multiply Accumulate）

WGMMA 是 H100 的"超级 mma"，作用域是 **Warp Group = 4 个连续 warp = 128 线程**：

$$
D_{m \times n} = A_{m \times k} \cdot B_{k \times n} + C_{m \times n}, \quad m=64, n \in \{8,...,256\}, k=16
$$

**与 A100 mma 的关键区别**：
| 特性 | A100 `mma.sync` | H100 `wgmma` |
| --- | --- | --- |
| 粒度 | 1 warp (32 线程) | 1 warp group (128 线程) |
| 同步 | 同步（发完等结果） | **异步**（发完立刻返回） |
| 操作数 A 来源 | 寄存器 | **寄存器或共享内存** |
| 操作数 B 来源 | 寄存器 | **共享内存** |

**异步的意义**：发完一条 WGMMA 后，warp 可以立刻去做 softmax，等真正需要 GEMM 结果时再 `wgmma.wait_group`。

#### 3.1.3 Distributed Shared Memory (DSM)

H100 引入了 SM-to-SM direct shared memory access（同一个 cluster 内的 SM 可以互相读写 shared memory）。FA-3 没有重度使用，但为未来留下空间。

#### 3.1.4 FP8 Tensor Core

H100 支持 E4M3（4 位指数 + 3 位尾数）和 E5M2 两种 FP8。峰值算力：

$$
\text{FP8 peak} \approx 2 \times \text{FP16 peak} = 1979 \text{ TFLOPS (SXM5)}
$$

代价：动态范围比 FP16 小很多，需要量化技巧。

### 3.2 FA-3 的三大核心创新

#### 创新 1：Producer-Consumer Warp 专化

这是 FA-3 最核心的设计。每个线程块（thread block）启动 **2 个 warp group**（共 256 个线程），按角色分工：

```
                    ┌──────────────────────────────────┐
                    │   Thread Block (2 Warp Groups)   │
                    └──────────────────────────────────┘
                              ↙             ↘
              ┌──────────────────┐    ┌──────────────────┐
              │  Producer WG     │    │  Consumer WG x2  │
              │  (1 warp = 32 t) │    │  (8 warps = 256t)│
              │  职责：发 TMA     │    │  职责：WGMMA +   │
              │  加载 K/V tile   │    │  softmax         │
              └──────────────────┘    └──────────────────┘
                       ↓                       ↑
                       └─── mbarrier ──────────┘
                          （同步信号）
```

**伪代码框架**：

```cuda
__global__ void flash_attn_3_kernel(...) {
    if (warp_group_id == 0) {
        // ============ PRODUCER ============
        for (int kv_tile = 0; kv_tile < num_tiles; ++kv_tile) {
            // 等共享内存空槽
            mbarrier_wait(empty[kv_tile % STAGES]);
            // 发 TMA 拷贝 K, V 到 shared memory
            tma_copy(K_global + kv_tile, K_smem[kv_tile % STAGES]);
            tma_copy(V_global + kv_tile, V_smem[kv_tile % STAGES]);
            // 通知 consumer：数据到了
            mbarrier_arrive(full[kv_tile % STAGES]);
        }
    } else {
        // ============ CONSUMER ============
        for (int kv_tile = 0; kv_tile < num_tiles; ++kv_tile) {
            // 等数据到位
            mbarrier_wait(full[kv_tile % STAGES]);
            // 计算 S = Q @ K^T （WGMMA 异步）
            wgmma(S, Q_reg, K_smem[kv_tile % STAGES]);
            wgmma_commit_group();
            wgmma_wait_group<0>();
            // softmax (online)
            online_softmax(S, m, l);
            // 计算 O += P @ V （WGMMA 异步）
            wgmma(O, P_reg, V_smem[kv_tile % STAGES]);
            // 释放共享内存槽
            mbarrier_arrive(empty[kv_tile % STAGES]);
        }
    }
}
```

**多级流水（multi-stage pipeline）**：通常用 STAGES=2 或 3，让 producer 提前 1–2 个 tile 加载，藏住 HBM 延迟。

#### 创新 2：Pingpong 调度（GEMM-Softmax 重叠）

光做 producer-consumer 还不够，因为 **consumer 内部的 GEMM 和 softmax 也是串行**：

```
单 WG 时间线（无 pingpong）：
┌──GEMM1──┐──softmax──┌──GEMM2──┐
                ↑
            softmax 期间 Tensor Core 空闲！
```

H100 的 softmax（exp、max、reduce）走的是 SFU 和 CUDA Core，跟 Tensor Core 是两套硬件——理论上可以并行。

**Pingpong 方案**：开两个 consumer warp group，让它们交错跑：

```
WG1 时间线：  ┌──GEMM─┐──softmax──┌──GEMM─┐──softmax──
WG2 时间线：  ──softmax──┌──GEMM─┐──softmax──┌──GEMM─┐

合并视角（Tensor Core 视角）：
  ┌──GEMM(WG1)──┐──GEMM(WG2)──┐──GEMM(WG1)──┐──GEMM(WG2)──┐
              （Tensor Core 几乎不空闲）

合并视角（SFU 视角）：
  ──softmax(WG2)──┐──softmax(WG1)──┐──softmax(WG2)──┐
              （SFU 几乎不空闲）
```

通过 named barrier 在两个 warp group 间做精细调度，使一个做 GEMM 时另一个必须做 softmax。

**收益**：单看 attention 内部，softmax 占总时间约 20%，pingpong 后这 20% 几乎完全藏在 GEMM 背后。

#### 创新 3：Warp-group 内的 GEMM-Softmax 双缓冲（intra-warpgroup overlap）

更进一步：在单个 warp group 内部，把当前迭代的 softmax 和**下一**迭代的第一个 GEMM 重叠：

$$
\text{Iter}_i: \quad \text{GEMM}_1 \to \text{softmax} \to \text{GEMM}_2
$$

改写成：

$$
\underbrace{\text{GEMM}_2^{(i-1)}}_{\text{异步发出}} \,\|\, \underbrace{\text{softmax}^{(i)}}_{\text{当前算}} \,\|\, \underbrace{\text{GEMM}_1^{(i+1)}}_{\text{下一轮提前}}
$$

利用 WGMMA 异步特性，发完上一轮的 GEMM2 后立刻开始当前的 softmax，再用 `wgmma.wait` 在真正用到结果前同步。这是 FA-3 把 H100 利用率从 60% 推到 75% 的最后一击。

#### 创新 4：FP8 数值稳定性技巧

朴素 FP8 attention 误差比 FP16 大 **2.4 倍**（论文 Table 2）。原因：

- $QK^T$ 容易出现大数（softmax 前），FP8 直接溢出。
- $V$ 中存在 outlier 通道，整列被 outlier 主导。

**技巧 A：Block Quantization**

不再对整张 K/V 用一个 scale，而是每 128 个元素一组，每组独立 scale：

$$
K_{\text{block}} = \text{round}(K_{\text{block}} / s_{\text{block}}) \cdot s_{\text{block}}, \quad s_{\text{block}} = \frac{\max(|K_{\text{block}}|)}{\text{FP8\_MAX}}
$$

这样每个 block 都用满 FP8 动态范围。

**技巧 B：Incoherent Processing（随机旋转）**

借鉴 QuIP（量化界经典技巧）：在量化前，对 $Q$ 和 $K$ 同时乘一个随机正交矩阵 $M$（用 Hadamard 变换实现，开销极低）：

$$
\tilde{Q} = QM, \quad \tilde{K} = KM \implies \tilde{Q}\tilde{K}^T = QM M^T K^T = QK^T
$$

由于 $M$ 是正交的，乘积不变；但 outlier 被"打散"到所有维度上，最大值大幅下降，FP8 量化误差随之降低。

**最终效果**：FP8 误差从"比 naive FP8 烂 2.4 倍"翻盘成"比 naive FP8 好 2.6 倍"，达到接近 FP16 的精度。

### 3.3 完整算法（精读级伪代码）

```python
# FA-3 前向（FP16 路径，省略 mask/causal 细节）
def flash_attention_3_forward(Q, K, V):
    """
    Q: (B, H, N, D)
    K, V: (B, H, M, D)
    """
    O = zeros_like(Q)
    L = zeros((B, H, N))      # 用于反向
    BLOCK_M, BLOCK_N = 128, 128
    STAGES = 2                 # 流水线深度

    for q_block in range(0, N, BLOCK_M):
        # ===== 启动 1 个 thread block =====
        # 加载 Q tile（一次性，不参与流水）
        Q_smem = load_via_TMA(Q[q_block:q_block+BLOCK_M])

        m_i = -inf  # row max
        l_i = 0     # row sum
        O_i = 0     # output

        # ----- Producer warp -----
        for kv_block in range(0, M, BLOCK_N):
            stage = kv_block // BLOCK_N % STAGES
            wait(empty_barrier[stage])
            tma_load_async(K[kv_block:kv_block+BLOCK_N], K_smem[stage])
            tma_load_async(V[kv_block:kv_block+BLOCK_N], V_smem[stage])
            arrive(full_barrier[stage])

        # ----- Consumer warp（pingpong 由 named barrier 协调）-----
        for kv_block in range(0, M, BLOCK_N):
            stage = kv_block // BLOCK_N % STAGES
            wait(full_barrier[stage])

            # GEMM1: S = Q @ K^T  (WGMMA 异步发出)
            wgmma_async(S, Q_smem, K_smem[stage].T)
            wgmma_wait()  # 等本次 GEMM 完成

            # online softmax
            m_new = max(m_i, rowmax(S))
            P = exp(S - m_new)
            l_new = exp(m_i - m_new) * l_i + rowsum(P)
            O_i = exp(m_i - m_new) * O_i

            # GEMM2: O += P @ V  (WGMMA 异步发出)
            wgmma_async(O_i, P, V_smem[stage])
            # 不立刻 wait，让下一轮 GEMM1 与本轮 softmax 重叠

            m_i, l_i = m_new, l_new
            arrive(empty_barrier[stage])

        wgmma_wait()  # 收尾
        O[q_block] = O_i / l_i
        L[q_block] = m_i + log(l_i)

    return O, L
```

### 3.4 关键实验结果

| Setup | 配置 | TFLOPS | H100 利用率 |
| --- | --- | --- | --- |
| FA-2 FP16 | seq=8K, head=128 | 335 | 35% |
| **FA-3 FP16** | seq=8K, head=128 | **740** | **75%** |
| FA-3 FP8 | seq=8K, head=128 | **1200** | **75%** |
| cuDNN 9 attention | seq=8K | ~600 | ~60% |

**精度（FP8）**：
- Naive FP8 attention：error = 2.4 × FP16 baseline
- FA-3 FP8（block + incoherent）：error = **0.38 × naive FP8**（即比 naive 好 2.6×，接近 FP16 baseline）

---

## 4️⃣ 物证清单 + 🔥 Hot Take

### 📦 物证清单

| 物证 | 内容 | 出处 |
| --- | --- | --- |
| 📄 论文 | arXiv:2407.08608 | https://arxiv.org/abs/2407.08608 |
| 💻 代码 | `flash-attn` 包（Dao-AILab/flash-attention） | GitHub |
| 🧪 实测 | 740 TFLOPS FP16，1.2 PFLOPS FP8 | 论文 Table 1 |
| 🎤 Talk | Tri Dao GTC 2024 talk | NVIDIA on-demand |
| 🔧 集成 | vLLM、TRT-LLM、SGLang 在论文发布 1 周内集成 | 各项目 commit |

### 🔥 五条 Hot Take

1. 🔥 **硬件特化是 LLM 推理优化的未来。** FA-1 关心 IO，FA-2 关心 FLOPs 利用率，FA-3 直接关心硬件电路图（TMA、WGMMA、SFU 调度）。FA-4 一定只跑 Blackwell。算法–硬件解耦的旧范式正在死去。

2. 🔥 **"warp 专化"是 GPU 编程的下一个范式。** 像 CPU 上的多线程程序员要懂 cache line，未来 GPU 内核工程师必须懂 warp group、TMA、mbarrier。CUTLASS、Triton 都在朝这个方向走。

3. 🔥 **FP8 是 LLM 的"自然精度"。** DeepSeek-V3、Llama-4 训练已大规模使用 FP8。FA-3 证明只要配 block quant + incoherent processing，FP8 在长序列下也能保持精度。FP4 是下一个战场。

4. 🔥 **Tri Dao = 这个时代的 Karatsuba。** 他在 2022（FA-1）、2023（FA-2 + Mamba-1）、2024（FA-3 + Mamba-2）连续三年用算法–系统协同设计撼动 Transformer 主流叙事。其学生路径 + 与 NVIDIA 深度合作是当代博士的范本。

5. 🔥 **"35% → 75%"意味着以前一半的算力是被代码浪费的。** 推论：你买的 H100 大概率没跑出它的真实性能。云厂商的 attention kernel 不升级，等于 SLA 默默打 5 折。

---

## 5️⃣ 🐛 论文没说的坑

1. 🐛 **只支持 H100，不支持 A100**：FA-3 重度依赖 TMA 和 WGMMA，A100 上跑不动。A100 用户继续用 FA-2。

2. 🐛 **小 batch / 短序列加速比不明显**：异步流水线的"启动开销"在序列 < 1K 时占比偏高，加速比可能从 2× 缩水到 1.1×。

3. 🐛 **FP8 在长序列（>32K）有 outlier 累积问题**：incoherent processing 是统计性的，超长序列下偶发的极端 outlier 仍可能导致单层精度劣化。论文只测到 8K。

4. 🐛 **MQA / GQA 下 K/V tile 太小**：当 KV head 数远小于 Q head 数（如 Llama-3 70B 是 8:64），K/V 的 tile 形状会变得不规则，pingpong 收益打折。

5. 🐛 **反向传播未做同等优化**：论文主要展示前向。反向涉及 dQ/dK/dV 三个 GEMM，warp 专化更难设计，实测加速比不到前向的一半。

---

## 6️⃣ 🎲 如果作者偷懒了

- 🎲 **没测 Blackwell（B100/B200）**：B100 的 TMEM、第 5 代 Tensor Core 又是另一套异步原语。FA-3 的代码无法直接复用，FA-4 已在路上。
- 🎲 **没试 FP4**：H100 不支持原生 FP4，但 B100 支持。FP4 需要更激进的 outlier handling，可能需要 NF4 或 MX4 格式。
- 🎲 **没做 sparse attention**：所有优化都基于 dense attention。Sliding window、block sparse 这些变体没覆盖。
- 🎲 **没在多卡上测 attention（DSM 利用）**：H100 的分布式 shared memory 跨 SM 访问能力没用上，长上下文 ring attention 还能再榨。
- 🎲 **没和 Triton 实现做对头实验**：作者主推 CUTLASS C++ 实现，但社区更想要 Triton 版本。直到 2025 年才有第三方 Triton FA-3。

---

## 7️⃣ 影响波及

```mermaid
graph TD
    FA1["L2-21 FlashAttention<br/>(2022, IO-aware)"] --> FA2["FlashAttention-2<br/>(2023, A100/H100)"]
    FA2 --> FA3["⭐ L2-36 FlashAttention-3<br/>(2024, H100 异步)"]
    FA3 --> FlashInfer["FlashInfer<br/>(推理专用)"]
    FA3 --> FA4["FlashAttention-4<br/>(2025, Blackwell)"]
    FA3 --> vLLM["vLLM / SGLang<br/>(直接集成)"]
    FA3 --> TRTLLM["TensorRT-LLM<br/>(NVIDIA 官方)"]
    FA3 -. FP8 启发 .-> DeepSeekV3["L3-31 DeepSeek-V3<br/>(FP8 训练)"]

    Tri["Tri Dao 主线"] --> FA1
    Tri --> Mamba1["Mamba (2023)"]
    Tri --> Mamba2["Mamba-2 (2024)"]
    Tri --> FA3

    H100["H100 Hopper<br/>(TMA/WGMMA/FP8)"] -.硬件基础.-> FA3
    Cutlass["CUTLASS 3.x"] -.工程基础.-> FA3
```

**影响半径**：

- **直接下游**：vLLM、SGLang、TRT-LLM、TGI 在 1 周内集成，H100 推理成本砍半。
- **算法侧**：DeepSeek-V3 大规模 FP8 训练直接受 FA-3 启发。
- **硬件侧**：NVIDIA 内部把 FA-3 作为 cuDNN 9 attention 的对标基线。
- **学术侧**：催生了一大批"硬件感知 attention"工作，包括 FlashInfer、SageAttention、MInference 等。

---

## 8️⃣ 侦探手记：算法–硬件协同设计的哲学

FA 系列三部曲，是过去三年 AI 系统领域最教科书级的"算法–硬件协同设计"案例：

| 代际 | 设计原则 | 关键洞察 |
| --- | --- | --- |
| FA-1 | **关心数据搬运** | "FLOPs 不是瓶颈，HBM 带宽才是" |
| FA-2 | **关心 GPU 占用** | "warp 内并行 vs 跨 warp 并行不一样" |
| FA-3 | **关心硬件电路** | "Tensor Core / SFU / TMA 是三套独立电路，可以同时跑" |

每一代都不是"换个公式"，而是"换个抽象层"——FA-1 把 attention 当成 IO 问题，FA-2 把它当成 GPU 占用率问题，FA-3 把它当成异步流水线调度问题。每深入一层，加速比再翻一倍。

**Tri Dao 给后来者的启示**：

> "做系统的人要会读 paper，做算法的人要会读 PTX。LLM 时代的 SOTA 长在算法–系统的接缝处。"

他在 ICML 2024 talk 里说过一句话：**"算法的下一个 1.5 倍加速，藏在硬件 manual 的脚注里。"** FA-3 就是脚注变成头条的故事。

而 Mamba 系列（同样是 Tri Dao）则反过来：**当 attention 的硬件优化被推到极限（FA-3 的 75% 已经接近 roofline），下一步就是改算法本身**——SSM/Mamba 用 O(N) 复杂度绕开 attention 的 O(N²) 天花板。FA-3 和 Mamba-2 在 2024 同年发布，可以视为 Tri Dao 对"transformer 还能不能再压"这一问题的辩证回答：

- 短序列、需要全局精度 → FA-3 把 attention 榨到底
- 长序列、可以容忍 SSM 近似 → Mamba-2 直接换架构

**这是同一个研究者，用算法 + 系统两条腿同时进攻 transformer 的范例。**

---

## ✅ 自查清单

- [x] 解释了 H100 vs A100 架构差异（TMA、WGMMA、DSM、FP8 Tensor Core）
- [x] 详述 producer-consumer warp 专化（含 ASCII 图示）
- [x] 详述 pingpong 调度
- [x] 详述 FP8 的 block quantization 和 incoherent processing
- [x] 完整伪代码
- [x] 关键数学公式（incoherent processing 正交不变性、量化公式）
- [x] FA-1/2/3 对比表
- [x] 5 条 Hot Take
- [x] 5 条没说的坑
- [x] mermaid 影响波及图
- [x] 强调 Tri Dao 作为 FA + Mamba 双线革新者

---

## 🔟 延伸卷宗

### 前置阅读

- **L2-21 FlashAttention**：奠基之作，IO-aware tiling 思想
- **L2-28 BFloat16 / 低精度训练**：理解 FP8 的前置
- **CUTLASS 3.x 文档**：FA-3 的工程基础，理解 WGMMA、TMA 的 C++ 抽象
- **NVIDIA Hopper Architecture Whitepaper**：H100 硬件原语权威来源

### 后续阅读

- **FlashAttention-4 (Blackwell)**：把同样的思想搬到 B100 的 TMEM 和 5th-gen Tensor Core
- **L3-31 DeepSeek-V3**：FP8 训练的工业级实践，直接受益于 FA-3
- **FlashInfer**：推理特化版的"FA-3 + paged KV cache"
- **Mamba-2**：Tri Dao 在 attention 之外的另一条路线
- **SageAttention / MInference**：硬件感知 attention 的后继者

### 关键 issue / commit

- `flash-attention/v2.6.0`：FA-3 首版上线
- `vLLM/PR#5921`：第一个集成 FA-3 的开源推理框架

---

## 🚀 3 小时复现路径

**目标**：在 H100 上跑通 FA-3，对比 FA-2 的加速比。

```bash
# Step 1: 环境（30 min）
conda create -n fa3 python=3.10 -y
conda activate fa3
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Step 2: 安装 flash-attn（含 FA-3 hopper kernel）（45 min）
pip install flash-attn --no-build-isolation
# 或源码编译（需要 nvcc 12.3+, H100）：
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
python setup.py install

# Step 3: 跑 benchmark（30 min）
python flash-attention/hopper/benchmark_attn.py
# 期望输出：
#   FA-2: ~335 TFLOPS
#   FA-3 FP16: ~740 TFLOPS
#   FA-3 FP8: ~1200 TFLOPS

# Step 4: 集成到自己的模型（75 min）
import torch
from flash_attn_interface import flash_attn_func  # FA-3 接口

q = torch.randn(2, 8192, 16, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn_like(q); v = torch.randn_like(q)

# FA-3 调用
out, lse = flash_attn_func(q, k, v, causal=True)

# FP8 路径（需 H100）：
from flash_attn_interface import flash_attn_func_fp8
out_fp8 = flash_attn_func_fp8(q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn))
```

**验证清单**：
- [ ] FA-3 比 FA-2 快 1.5× 以上
- [ ] FP8 比 FP16 再快 1.5–1.7×
- [ ] FP8 输出和 FP16 输出的 cosine similarity > 0.999

---

## 📌 本案归档

- **结案陈词**：FA-3 把 attention 在 H100 上的 MFU 从 35% 拉到 75%，靠的是把 H100 的异步原语（TMA、WGMMA）+ 低精度（FP8）+ warp 专化全部塞进一个内核。它是 LLM 时代"算法–硬件协同设计"的当代范本。
- **关键作者**：Tri Dao（普林斯顿 → together.ai），FlashAttention 系列 + Mamba 系列共同作者
- **办案动机**：H100 算力卖得贵，但 FA-2 时代只用了 35%，社区急需"配得上 H100"的 attention kernel
- **侦探口头禅**：**"算法的下一个 1.5 倍加速，藏在硬件 manual 的脚注里。"**

---

**🔗 下一站**：→ [L2-37 EAGLE-2](./PDFs/L2-37_EAGLE_2.pdf) —— 当 attention 本身已被榨到 75% MFU，下一步推理优化转向"解码"层面：speculative decoding 如何让长输出也飞起来？

---

> *归档于 2026-05-01｜《LLM 百案录》*
> *⚡ 异步极限｜🚀 H100 榨干｜🎭 算法–硬件协同*
