# ➗ 案件 L2-34：Differential Transformer — 用"差分注意力"消灭噪声头

> **《LLM 百案录》第 034 案 · 差分注意力**
> *2024 年 10 月 7 日，微软研究院抛出一个让整个 transformer 圈瞠目的论断：*
> *"标准 attention 把太多注意力分配给了无关 token——我们用两组 softmax 的差，把噪声直接消掉。"*
> *论文标题就一个数学符号：**Diff Transformer**。*
> *差分电路的物理学思想第一次走进 attention 设计。*

🚇 [30秒速览](#速览) ｜ 🚲 [3分钟通读](#通读) ｜ 🚗 [30分钟精读](#精读) ｜ 🚀 [3小时复现](#复现)

🎭 **叙事母题**：➗ **差分注意力** —— 两组 attention 相减，共模噪声自然抵消

---

## 0️⃣ 案件档案

| 案情要素 | 内容 |
|---|---|
| **案发时间** | 2024-10-07（Ye et al.，[arXiv 2410.05258](https://arxiv.org/abs/2410.05258)） |
| **嫌疑人** | Tianzhu Ye、Li Dong、Yuqing Xia、Yutao Sun、Yi Zhu、Gao Huang、Furu Wei |
| **作案地点** | Microsoft Research + Tsinghua + UC Berkeley |
| **受害者** | 标准 softmax attention 的"注意力涣散"现象（attention sinks + 无关 token 占比过高） |
| **作案凶器** | $\text{DiffAttn}(Q_1, K_1, Q_2, K_2, V) = (\text{softmax}(Q_1 K_1^T) - \lambda \cdot \text{softmax}(Q_2 K_2^T)) V$，可学习的 $\lambda$ |
| **作案动机** | "差分放大器在电路里抑制共模噪声——transformer 的 attention 噪声能不能用同样思路？" |
| **结案陈词** | DIFF-3B 在 1T tokens 上预训练，与同尺寸 Transformer 比 ~65% 参数即达同 perplexity；长上下文 NIAH 显著优于 Transformer；激活异常值大幅减少，量化更友好 |

### 五维雷达

| 维度 | 评分 | 解释 |
|---|---|---|
| 创新性 | **9/10** | ← "两个 softmax 相减" 简单到令人怀疑，但效果显著 |
| 影响力 | **8/10** | ← 引爆 attention 重设计浪潮，多个后续工作跟进 |
| 复杂度 | **3/10** | ← 修改 attention 一处就能跑 |
| 可复现 | **9/10** | ← 微软开源代码 + HuggingFace 集成 |
| 争议度 | **6/10** | ← "为什么相减就 work" 缺少彻底理论解释 |

### 精确事实卡

| 字段 | 精确值 | 来源 |
|---|---|---|
| 主测模型 | 830M / 1.4B / **2.8B** / 6.8B | 论文 §3 |
| 训练数据 | 1T tokens（StarCoderData + RefinedWeb 等） | §3.1 |
| 头数减半 | DIFF 用 h/2 头双倍，与 Transformer 同总头数 | §2.2 |
| λ 初始化 | $\lambda_{\text{init}} = 0.8 - 0.6 \exp(-0.3 \cdot l)$（l 是层号） | §2.3 |
| 65% 参数等效 | DIFF-1.4B perplexity ≈ Transformer-2.0B | Fig 5 |
| NIAH 4K | DIFF 89% vs Transformer 71% | Table 4 |
| Activation Outliers | DIFF 显著少于 Transformer | Fig 9 |
| 训练硬件 | 64 × A100 | §3.1 |

---

## 1️⃣ <a id="速览"></a>🚇 30 秒速览

> **一句话案情**：把每个 head 的 Q、K 拆成两组，分别 softmax 注意力 $A_1, A_2$，最终注意力 = $A_1 - \lambda A_2$。**共模噪声**（attention sinks、对所有 token 的均匀关注）会被相减抵消，**真正的目标 token 信号**被保留。

- **核心**：从"单 softmax 注意力"升级到"差分双 softmax"。
- **关键参数**：可学习标量 $\lambda$ 控制减法强度。
- **效果**：65% 参数即可对标同等 perplexity 的标准 Transformer。
- **副产品**：activation outliers 大幅减少，4-bit 量化几乎无损。

---

## 2️⃣ <a id="通读"></a>🚲 3 分钟通读：为什么是 DIFF Transformer（Why）

### 时代背景：attention 的"噪声"问题

```text
2017  原始 Transformer       softmax 强迫所有概率和为 1
2023  Attention Sinks         发现 BOS token 吸引大量无关注意力
2024  StreamingLLM             发现 sink 普遍存在
2024-10  Diff Transformer     ← 主动消除 sink 噪声
```

### 三个动机

```python
# 动机 1：softmax 总和 = 1 的诅咒
# 即使没相关 token，softmax 还是要分配概率
# → 注意力被"稀释"到无关位置

# 动机 2：电路里的差分放大器思想
# differential amplifier = (V+ - V-) × gain
# 共模噪声（V+ 和 V- 同时叠加的）被抵消
# 仅差分信号被放大

# 动机 3：长上下文中信号 / 噪声 比恶化
# 上下文越长，噪声 token 越多
# 真正有用的 needle 容易被淹没
```

### DIFF 的灵感

> 把电路里的 differential amplifier 移植到 attention：用两组 attention map 当 V+ 和 V-，相减消共模。

---

## 3️⃣ <a id="精读"></a>🚗 30 分钟精读

### 3.1 数学定义（论文 §2）

#### 标准 attention

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

#### Diff Attention

$$Q_1, Q_2 = \text{split}(Q), \quad K_1, K_2 = \text{split}(K)$$

$$\text{DiffAttn}(Q, K, V, \lambda) = \left(\text{softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \cdot \text{softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)\right) V$$

> **关键**：两个 softmax 是独立的概率分布，但相减后结果**不是**概率分布。这是 DIFF 与标准 attn 最大的不同。

### 3.2 λ 的设计

#### 可学习 λ：rank-1 重参数化

$$\lambda = \exp(\lambda_{q1} \cdot \lambda_{k1}) - \exp(\lambda_{q2} \cdot \lambda_{k2}) + \lambda_{\text{init}}$$

其中 $\lambda_{q1}, \lambda_{k1}, \lambda_{q2}, \lambda_{k2}$ 是 d-维可学习向量。

#### 层级初始化

$$\lambda_{\text{init}} = 0.8 - 0.6 \exp(-0.3 \cdot l)$$

- 浅层（l 小）：$\lambda_{\text{init}} \approx 0.34$
- 深层（l 大）：$\lambda_{\text{init}} \approx 0.8$

> **侦探洞察**：深层需要更强的差分（噪声更多），浅层差分轻一些。这个 schedule 是经验性的，论文做了消融。

### 3.3 头数补偿

| 模型 | n_heads | head_dim |
|---|---|---|
| 标准 Transformer | 32 | 128 |
| **DIFF Transformer** | **16** | **128**（每头算两组 Q1/Q2、K1/K2） |

DIFF 把头数减半（每头执行差分），保持总参数和计算量与标准 transformer 相当。

### 3.4 PyTorch 关键实现

```python
class DiffAttention(nn.Module):
    def __init__(self, d, n_heads, layer_idx):
        super().__init__()
        self.d, self.n_heads = d, n_heads
        self.dh = d // n_heads
        # 注意：每头需要两组 Q/K，所以投影维度是 2*dh
        self.q_proj = nn.Linear(d, 2*d)  # 双倍 Q
        self.k_proj = nn.Linear(d, 2*d)  # 双倍 K
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        
        # λ 可学习参数（rank-1）
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.dh).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.dh).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.dh).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.dh).normal_(0, 0.1))
    
    def forward(self, x):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, 2, self.dh)
        K = self.k_proj(x).view(B, T, self.n_heads, 2, self.dh)
        V = self.v_proj(x).view(B, T, self.n_heads, self.dh)
        
        Q1, Q2 = Q.unbind(-2)  # split 成 2 组
        K1, K2 = K.unbind(-2)
        
        # 两组 attention
        attn1 = F.softmax(torch.einsum("bthd,bshd->bhts", Q1, K1) / (self.dh ** 0.5), dim=-1)
        attn2 = F.softmax(torch.einsum("bthd,bshd->bhts", Q2, K2) / (self.dh ** 0.5), dim=-1)
        
        # λ 计算
        lam = (torch.exp(self.lambda_q1 @ self.lambda_k1) 
               - torch.exp(self.lambda_q2 @ self.lambda_k2) 
               + self.lambda_init)
        
        # 差分
        attn = attn1 - lam * attn2
        
        out = torch.einsum("bhts,bshd->bthd", attn, V).reshape(B, T, -1)
        return self.o_proj(out)
```

### 3.5 为什么差分能消噪？（论文 §2.4 分析）

```text
标准 attention 的两种成分：
- 信号成分：query 与 key 真正相关时的高分
- 噪声成分：所有 token 的"基线" attention（含 sink）

差分 attention 的假设：
- 第一组 (Q1, K1) 学到 "信号 + 噪声"
- 第二组 (Q2, K2) 学到 "噪声"（共模）
- 相减 → 仅留信号

实证：
- 论文 Fig 7：DIFF 的 attention map 比 Transformer 干净许多
- 1T pretraining 后，DIFF 的 attention sink 比例 < 5% (vs T 的 ~30%)
```

> **侦探洞察**：DIFF 没有显式监督"哪部分是噪声"，但**两组 softmax 的减法在端到端训练下自然分化**为"信号头"和"噪声头"。这是涌现行为。

---

## 4️⃣ 物证清单（Results）& 🔥 Hot Take

### 主结果（论文 Table 1，1T tokens 预训练）

| Model | Params | Wikitext PPL ↓ | C4 PPL | LAMBADA |
|---|---|---|---|---|
| Transformer-1.4B | 1.4B | 14.40 | 11.12 | 65.5 |
| **DIFF-1.4B** | **1.4B** | **13.84** | **10.65** | **67.8** |
| Transformer-2.8B | 2.8B | 12.55 | 9.53 | 71.2 |
| **DIFF-2.8B** | **2.8B** | **12.10** | **9.20** | **73.4** |

### 长上下文 NIAH（论文 Table 4）

| Length | Transformer-2.8B | DIFF-2.8B |
|---|---|---|
| 1K | 100% | 100% |
| 2K | 95% | 100% |
| 4K | 71% | **89%** |
| 8K | 38% | **62%** |

### Activation Outliers（论文 Fig 9）

| Model | Max activation | Outlier 比例 |
|---|---|---|
| Transformer-2.8B | 318 | 高 |
| **DIFF-2.8B** | **9.4** | **极低** |

→ 4-bit 量化几乎无损，Transformer 4-bit 掉 5+ PPL。

### 🔥 Hot Take

1. **"减法"在 deep learning 历来低估** —— ResNet 的核心是 + 不是 -，所有 attention 设计也都是 +。DIFF 第一次正经把"-"做主角。

2. **65% 参数对等是真省钱** —— 训 1.4B 拿到 2B 效果，对开源社区是直接的成本福利。

3. **量化友好性是隐藏价值** —— activation outliers 锐减，意味着 GPTQ / AWQ 量化 DIFF 模型几乎免调。这点对 deployment 极有意义。

4. **理论解释仍欠债** —— 论文给了"差分放大器"类比，但缺少形式化证明。**为什么两个独立的 softmax 一定能分化为信号 vs 噪声**？这是开放问题。

5. **生产化阻力**：要从头预训练，无法 retrofit 已有模型。这是 DIFF 商业落地的最大障碍。

---

## 5️⃣ 🐛 论文没说的坑

1. **不能直接 retrofit Llama** —— 把 Llama-7B 的 attention 改成 DIFF 必须重新预训练，几乎等同从零训。

2. **λ 初始化敏感** —— 不按 schedule 而用固定 0.5 会导致前期不稳定。

3. **FlashAttention 不直接支持** —— 因为是两个独立 softmax 后相减，需要自定义 kernel。社区 DIFF-FlashAttn 在 2024-12 才出来。

4. **训练慢 ~10%** —— 比标准 Transformer 多一组 Q/K 计算，前向计算量增加。

5. **小模型增益较小** —— 600M 以下 DIFF 与 Transformer 几乎打平。**1B+ 才能看到明显差异**。

---

## 6️⃣ 🎲 如果作者偷懒了（如果做更多）

### 实验维度

- **更大规模**：7B、70B 上 DIFF 是否仍 65% 参数对等？
- **MoE 适配**：DIFF + MoE 的协同效应。
- **多模态**：在 ViT、视觉 transformer 上 DIFF 是否同样 work？

### 理论维度

- **形式化噪声分解**：能否证明 DIFF 必然分化为 signal vs noise 两组？
- **与 attention sink 的关系**：DIFF 是否本质上学到了 "subtract sink"？

### 应用维度

- **替换式部署**：能否冻结 V 矩阵、只 fine-tune Q/K 双组+λ，把 Llama 改造成 DIFF？
- **Inference 优化**：差分 attention 的专用 CUDA kernel。

---

## 7️⃣ 影响波及

```mermaid
graph TD
    ATTN[Attention<br/>2017] --> SINK[Attention Sinks<br/>2023]
    SINK --> STREAM[StreamingLLM<br/>2024]
    
    ATTN --> DIFF[Diff Transformer L2-34<br/>2024-10]
    
    DIFF --> DIFFFLASH[DIFF-FlashAttn<br/>2024-12]
    DIFF --> QUANT[量化友好<br/>4-bit 无损]
    DIFF --> NIAH_BETTER[长上下文检索增强]
    
    DIFF --> LATER[2025+ 后续工作:<br/>Selective Attn / NSA]
    
    style DIFF fill:#ffd700,stroke:#333,stroke-width:3px
    style QUANT fill:#90ee90
```

DIFF 的真正影响**不止 perplexity**，而在它**激发了"主动消噪"派系**——后续 Selective Attention、NSA 都受其启发。

---

## 8️⃣ 侦探手记

读完 DIFF Transformer，我盯着自家 LLM 的 attention map 可视化沉思。

第一感受是**敬意**。Furu Wei 团队（同组也做了 LongNet、RetNet）连续几年在 attention 设计上做"反主流"创新。**最简单的修改往往最深刻**——一个减号，干掉了 7 年的 attention 涣散问题。

第二感受是**审视**。DIFF 真的在"消噪"吗？还是仅仅在做更复杂的非线性？这点论文没给彻底证明。**机理解释仍是黑箱**。

第三感受是**期待**。DIFF + MLA + MoE 三件套，可能是 2026 年下一代基座的标配。每个都解决一个具体问题：**MLA 省 KV，MoE 省计算，DIFF 省噪声**。三者正交，可叠加。

> 案件结案。下一站：Medusa 用多头并行解码加速推理。

---

## 自查清单

- ✅ 通读论文 28 页
- ✅ 实现 DiffAttention 玩具版（170M 模型）
- ✅ 在 Wikitext-103 上对比 PPL
- ✅ 可视化 DIFF vs Transformer attention map
- ❌ 未在 1T 预训练数据上复现（计算太贵）
- ❌ 未做 INT4 量化对比

---

## 🔟 延伸卷宗

### 前置依赖

- 📚 [L1-01 Attention Is All You Need](./L1-01_Attention_Is_All_You_Need.md)
- 📚 [L2-21 FlashAttention](./L2-21_FlashAttention.md)
- 📚 [L4-11 LM-Infinite](./L4-11_LM_Infinite.md)（attention sinks）

### 后续推荐

- 🎯 Selective Attention（2024-11，类似消噪思想）
- 🎯 NSA / Native Sparse Attention（DeepSeek 2025-02）
- 🎯 [L2-33 DeepSeek-V2 MLA](./L2-33_DeepSeek_V2_MLA.md)（KV 压缩对照）

### 相关资源

- 📦 GitHub: [microsoft/unilm/Diff-Transformer](https://github.com/microsoft/unilm)
- 🤗 HuggingFace: [microsoft/Diff-Transformer-2.8B](https://huggingface.co/microsoft)
- 📄 arXiv: [2410.05258](https://arxiv.org/abs/2410.05258)

### 🚀 <a id="复现"></a>3 小时复现路径

#### Step 1：安装（10 分钟）

```bash
git clone https://github.com/microsoft/unilm.git
cd unilm/Diff-Transformer
pip install -e .
```

#### Step 2：DiffAttention 玩具实现（30 分钟）

参考上文 §3.4 完整 PyTorch 实现，单元测试：

```python
diff = DiffAttention(d=512, n_heads=8, layer_idx=4).cuda()
x = torch.randn(2, 1024, 512).cuda()
out = diff(x)
assert out.shape == x.shape
```

#### Step 3：训练 60M 玩具模型（90 分钟）

```bash
python train.py \
    --model_config configs/diff_60m.json \
    --data wikitext-103-v1 \
    --batch_size 32 \
    --lr 6e-4 \
    --num_steps 50000 \
    --output_dir ./diff-60m
```

#### Step 4：与 Transformer baseline 对比（30 分钟）

```bash
# 同样配置，attention 换回普通
python train.py --model_config configs/transformer_60m.json ...
# 比较 PPL
python eval.py --model ./diff-60m
python eval.py --model ./transformer-60m
```

预期：DIFF PPL 比 Transformer 低 ~3%。

#### Step 5：可视化 attention（30 分钟）

```python
import matplotlib.pyplot as plt
attn1, attn2, attn_diff = model.get_attention_maps(text)
fig, axes = plt.subplots(1, 3)
for ax, m, t in zip(axes, [attn1, attn2, attn_diff], ["Branch1","Branch2","Diff"]):
    ax.imshow(m); ax.set_title(t)
plt.savefig("attn_compare.png")
```

#### Step 6：4-bit 量化测试（30 分钟）

```bash
python quantize.py --model ./diff-60m --bits 4 --method gptq
python eval.py --model ./diff-60m-int4
```

预期：DIFF int4 PPL 与 fp16 几乎一致；Transformer int4 显著退化。

---

## 📌 本案归档

| 字段 | 值 |
|---|---|
| 案件编号 | L2-34 |
| 笔记版本 | v1「差分注意力版」 |
| 叙事母题 | ➗ 差分注意力 |
| 推荐指数 | ⭐⭐⭐⭐ |
| 学习时长 | 30s / 3min / 30min / 3h |
| 上次更新 | 2026-05-02 |
| 关联案件 | L1-01 (Attention)、L2-33 (MLA)、L4-11 (Sinks) |
| 上一站 | ← [L2-33 DeepSeek-V2](./L2-33_DeepSeek_V2_MLA.md) |
| 下一站 | → [L2-35 Medusa](./L2-35_Medusa.md) |

---

> *"加法堆出复杂度，减法剥出本质。Attention 也不例外。"*
> *—— 侦探的结案陈词，2026 年春于 LLM 百案录*
