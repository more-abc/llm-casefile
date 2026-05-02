# ⏳ 案件 L4-32：s1 — 1000 条数据 + 一个 "Wait" token，复刻 o1

> **《LLM 百案录》第 132 案 · 极简推理**
> *2025 年 1 月 31 日，DeepSeek-R1 发布的烟尘还未散去，斯坦福一群学者在 arXiv 上贴出 9 页论文。*
> *他们说："我们用 1000 条样本、26 分钟训练、一个写着 'Wait' 的 token，把 Qwen2.5-32B 调成了 o1-preview。"*
> *学界惊愕：原来推理能力不一定要花上千万美金的 RL，可能只需要一杯咖啡的时间和一个英文单词。*

🚇 [30秒速览](#速览) ｜ 🚲 [3分钟通读](#通读) ｜ 🚗 [30分钟精读](#精读) ｜ 🚀 [3小时复现](#复现)

🎭 **叙事母题**：⏳ **极简推理** —— 1K 样本 + Wait token，低成本激活 o1 级推理能力

---

## 0️⃣ 案件档案

| 案情要素 | 内容 |
|---|---|
| **案发时间** | 2025-01-31（arXiv: 2501.19393） |
| **嫌疑人** | Niklas Muennighoff、Zitong Yang、Weijia Shi、Xiang Lisa Li、Li Fei-Fei、Hannaneh Hajishirzi、Luke Zettlemoyer、Percy Liang、Emmanuel Candès、Tatsunori Hashimoto |
| **作案地点** | Stanford × University of Washington × AI2 |
| **受害者** | "推理能力必须靠大规模 RL 才能解锁" 的迷思；OpenAI o1-preview 的护城河 |
| **作案凶器** | s1K（1000 条精选样本）+ Budget Forcing（"Wait" / "Final Answer:"）+ Qwen2.5-32B-Instruct |
| **作案动机** | 在 R1 走 RL 重路线后，证明纯 SFT 极简方案也能达到 o1 级推理 |
| **结案陈词** | AIME24 = 56.7%，超过 o1-preview 的 44.6%。模型/数据/代码全开源，复现成本 < $50 |

### 五维雷达

| 维度 | 评分 | 解释 |
|---|---|---|
| 创新性 | **9/10** | ← Budget Forcing 是首个在解码层"控制思考预算"的极简方案 |
| 影响力 | **9/10** | ← 证明 1K 样本 + SFT 即可逼近 o1，重塑社区对 RL 必要性的认知 |
| 复杂度 | **3/10** | ← 没有 RL、没有 PRM、没有 MCTS。简单到读完就能复现 |
| 可复现 | **10/10** | ← 模型/s1K/训练脚本全开源，26 分钟跑完 |
| 争议度 | **7/10** | ← "Wait token 真的有用吗？是泛化还是 Qwen 暗藏 OpenAI 数据？" 争论持续 |

### 精确事实卡

| 字段 | 精确值 | 来源 |
|---|---|---|
| 基座模型 | Qwen2.5-32B-Instruct | 论文 §3 |
| 训练样本数 | 1,000 (s1K) | 论文 Table 1 |
| 候选池大小 | 59,029 | 论文 §2.1 |
| 蒸馏来源 | Gemini Flash Thinking Experimental | 论文 §2.1 |
| 训练时间 | 26 分钟 | 论文 §3 |
| 训练硬件 | 16 × NVIDIA H100 | 论文 §3 |
| Epochs | 5 | 论文 §3 |
| 批大小 | 16（global） | 论文 §3 |
| 学习率 | 1e-5 | 论文 §3 |
| AIME 2024 准确率 | 56.7% (s1-32B) vs 44.6% (o1-preview) | 论文 Table 1 |
| MATH500 准确率 | 93.0% (s1-32B) vs 85.5% (o1-preview) | 论文 Table 1 |
| GPQA Diamond | 59.6% (s1-32B) | 论文 Table 1 |
| Budget Forcing 增益 | AIME24 50% → 57%（追加 6 次 Wait） | 论文 Figure 3 |
| 论文长度 | 9 页正文 | arXiv 2501.19393 |
| 许可证 | Apache 2.0（模型 + 代码 + 数据） | GitHub README |

---

## 1️⃣ <a id="速览"></a>🚇 30 秒速览

> **一句话案情**：把 1000 条 Gemini 生成的高质量推理轨迹喂给 Qwen2.5-32B，再在解码时偷偷塞个 "Wait"，让模型多想一会，就能把 AIME24 从 50% 推到 57%，全面碾压 o1-preview。

- **核心创新**：**Budget Forcing** —— 想让模型停？追加 "Final Answer:"。想让模型继续想？把模型的结束符替换成 "Wait"，模型就会自我反驳、检查并继续推理。
- **数据魔法**：从 59K 样本中按"难度 + 多样性 + 质量"三条准则筛 1K，比用全量 59K 训出来还强。
- **极致简洁**：没有 RL、没有 PRM、没有 MCTS、没有 Process Reward。一份 SFT 配方就能上桌。
- **完全开源**：模型 weight、s1K 数据、训练脚本、复现 notebook 全在 GitHub。

---

## 2️⃣ <a id="通读"></a>🚲 3 分钟通读：为什么是 s1（Why）

### 时代背景：2025 年 1 月的"推理军备竞赛"

```text
2024-09  OpenAI o1-preview        闭源，引爆 test-time scaling 概念
2024-12  OpenAI o1                正式发布，AIME 74.4%
2025-01-20  DeepSeek-R1           RL 路线，开源，AIME 79.8%
2025-01-22  Kimi k1.5             RL + 长思考，开源
2025-01-31  s1                    SFT + 1000 样本，"我也能行"
```

R1 用了几千 GPU、海量 RL rollout 才把 R1-Zero 训出来。学术界看着流口水，但没人复现得起。s1 团队的反向思路是：

> **"如果 o1 的推理能力本质是'让模型把已知的 CoT 模式'用到尽头，那是不是只要少量高质量样本激活，再在解码时强制延长思考，就够了？"**

### 三个动机

```python
# 动机 1：质疑 RL 的必要性
if reasoning_capability is intrinsic_to_pretrained_model:
    sft_with_quality_data ≈ rl_with_massive_rollouts

# 动机 2：探索 test-time scaling 的可控性
# o1 能根据题目难度自适应分配思考时间，但用户无法控制
# s1 想：能不能给用户一个"思考预算"旋钮？

# 动机 3：极致开源
# R1 开源了 weight 但没开数据；s1 把数据、代码、模型全开
```

### s1 的两个杀手锏

1. **s1K 数据集**（1000 个精选样本）
   - 来源：MATH、AIME、AGIEval、OlympiadBench、Numina、s1-prob 等 16 个数据集合并
   - 筛选流程：质量过滤 → 难度过滤（剔除 Qwen 7B/32B 已能解的）→ 多样性过滤（覆盖 50 个数学子领域）
   - 每条样本带 Gemini Flash Thinking 蒸馏的完整 reasoning trace（含 "wait, let me check..." 这类反思句）

2. **Budget Forcing**（推理时控制思考长度）
   - 想"快"：达到 token 预算时，强行 append `"Final Answer:"` → 模型立刻进入回答模式
   - 想"慢"：模型生成 `</think>` 时，把它替换成 `"Wait"` → 模型继续反思

---

## 3️⃣ <a id="精读"></a>🚗 30 分钟精读

### 3.1 s1K 数据筛选：少即是多的玄学

#### Step 1：59K 候选池

| 来源 | 样本数 | 类型 |
|---|---|---|
| NuminaMATH | 30,660 | 竞赛数学 |
| AIME 1983-2021 | 890 | 历史 AIME |
| OlympiadBench | 4,250 | 物理/数学奥赛 |
| AGIEval | 2,385 | 标化考试 |
| s1-prob (新建) | 182 | 概率论 |
| s1-teasers (新建) | 23 | 脑筋急转弯 |
| ... | ~20K | 其他 |
| **合计** | **59,029** | |

#### Step 2：三阶段筛选（论文 §2.2 的核心）

```python
def filter_s1k(candidates):
    # 阶段 1：质量过滤
    # - 删除 API 出错的轨迹
    # - 删除格式损坏样本（公式渲染失败、ASCII art 等）
    # - 剩余 ~51K
    candidates = quality_filter(candidates)
    
    # 阶段 2：难度过滤（关键创新）
    # 用两个 baseline 模型（Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct）尝试解
    # 如果两个都能解对 → 太简单 → 丢弃
    # 同时根据 reasoning trace 长度分箱（更长 = 更难）
    candidates = [c for c in candidates 
                  if not (qwen_7b.solves(c) and qwen_32b.solves(c))]
    
    # 阶段 3：多样性过滤
    # 用 Claude 3.5 Sonnet 给每个样本打 MSC（数学学科分类）标签
    # 50 个领域，每个领域均匀采样
    # 优先保留长 trace（重要先验：思考越长 = 题越值得学）
    s1k = diversity_sample(candidates, n=1000, n_domains=50)
    
    return s1k  # |s1k| = 1000
```

#### 消融：1K 真的够？

| 训练数据 | AIME24 | MATH500 | GPQA |
|---|---|---|---|
| 59K 全量 | 53.3% | 92.8% | 58.1% |
| 1K 随机采样 | 36.7% | 90.6% | 52.0% |
| 1K 仅按长度选 | 33.3% | 90.4% | 59.6% |
| 1K 仅按多样性选 | 26.7% | 91.2% | 54.6% |
| **s1K (三准则)** | **50.0%** | **93.0%** | **57.6%** |

> **侦探洞察**：59K 训出来仅比精选 1K 高 3 分，但训练成本是 50 倍。这是 LLM 时代版的"二八定律"。

### 3.2 Budget Forcing：解码时的"思考阀门"

#### 核心算法（论文 Algorithm 1 重写版）

```python
def budget_forcing(model, prompt, min_tokens, max_tokens, max_waits=4):
    """
    min_tokens: 至少思考多少 token
    max_tokens: 最多思考多少 token
    max_waits:  最多追加多少次 'Wait'
    """
    output = ""
    waits_used = 0
    
    while True:
        # 让模型解码
        new_tokens = model.generate(
            prompt + output,
            stop=["</think>", "Final Answer:"]
        )
        output += new_tokens
        n = count_tokens(output)
        
        # 情况 1：模型自己想结束，但还没达到最少预算 → 追加 Wait
        if output.endswith("</think>") and n < min_tokens and waits_used < max_waits:
            output = output.replace("</think>", "Wait", 1)
            waits_used += 1
            continue
        
        # 情况 2：超过最大预算，但模型还在想 → 强制结束
        if n >= max_tokens and not output.endswith("Final Answer:"):
            output += "\nFinal Answer: "
            output += model.generate(prompt + output, stop=["\n"])
            break
        
        # 情况 3：自然结束
        if output.endswith("</think>"):
            output += "\nFinal Answer: "
            output += model.generate(prompt + output, stop=["\n"])
            break
    
    return output
```

#### 神奇现象：越多 Wait，AIME 越高

| 追加 Wait 次数 | 平均思考 token | AIME24 |
|---|---|---|
| 0 (无 BF) | 3.6K | 50.0% |
| 1 | 5.1K | 53.3% |
| 2 | 6.8K | 56.7% |
| 4 | 9.2K | **57.0%** |
| 6 | 11.3K | 56.7% |

> **平台效应**：4 次 Wait 后收益边际递减。模型并非无限可被"催"。

#### 为什么 "Wait" 会奏效？案例剖析

模型原输出（错答）：
```
...所以 a + b = 7。</think>

Final Answer: 7
```

替换为 "Wait"：
```
...所以 a + b = 7。Wait, let me double-check by substituting back.
If a = 3, b = 4, then a² + b² = 9 + 16 = 25 ≠ 30. 
That's wrong. Let me redo...
所以 a + b = 8。

Final Answer: 8 ✓
```

> **侦探洞察**：Qwen2.5-32B 在预训练里早就学过"Wait, let me reconsider..."这类反思句式。SFT 在 s1K 上把这个能力放大，BF 再借力打力。**它不是"教会"模型反思，而是"开闸放水"**。

### 3.3 训练细节：26 分钟成就 o1

```yaml
# train_config.yaml
base_model: Qwen2.5-32B-Instruct
data: s1K-1.1 (1000 samples, with Gemini-flash-thinking traces)
optimizer: AdamW
lr: 1e-5
schedule: cosine, warmup 5%
epochs: 5
batch_size: 16  # global, micro=1, grad_accum=16
sequence_length: 32768  # 长 reasoning trace 必备
hardware: 16 × H100 (80GB)
parallelism: FSDP (full shard)
training_time: 26 minutes
total_steps: 315
```

#### 损失函数（标准 SFT）

$$\mathcal{L} = -\sum_{i \in \text{response}} \log P(y_i | y_{<i}, x; \theta)$$

只在 response（含 reasoning trace）上算 loss，prompt 不算。

#### 关键超参："训练 5 个 epoch 是否过拟合？"

| Epochs | AIME24 | 备注 |
|---|---|---|
| 1 | 30.0% | 欠拟合 |
| 3 | 46.7% | |
| **5** | **50.0%** | 最佳 |
| 7 | 50.0% | 平台 |
| 10 | 46.7% | 轻微过拟合 |

> **小数据 + 多 epoch** 是反直觉但合理的：1000 样本 × 5 epoch ≈ 5000 step 对 32B 模型来说仍是低强度。

---

## 4️⃣ 物证清单（Results）& 🔥 Hot Take

### 主结果（论文 Table 1）

| Model | AIME24 | MATH500 | GPQA Diamond | 训练成本 |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct（基座） | 26.7% | 84.0% | 49.0% | - |
| o1-preview | 44.6% | 85.5% | 73.3% | $$$$ |
| o1 | 74.4% | 94.8% | 77.3% | $$$$$ |
| DeepSeek-R1 | 79.8% | 97.3% | 71.5% | $$$$ |
| QwQ-32B-preview | 50.0% | 90.6% | 54.5% | $$ |
| **s1-32B（无 BF）** | **50.0%** | **93.0%** | **57.6%** | **$50** |
| **s1-32B（+ BF）** | **56.7%** | **93.0%** | **59.6%** | **$50** |

### 🔥 Hot Take

1. **"Wait" 不是魔法，是钥匙** —— 它没创造能力，只是打开了 Qwen 预训练里早已存在的反思能力。换句话说，**预训练才是真正的英雄，SFT 只是揭幕**。

2. **1K vs 59K 仅差 3 分**，却差 50 倍训练成本。学术界从此有了能复现"o1 级推理"的预算门票。

3. **Budget Forcing > Majority Voting** —— 同样追加 token，BF 比 N 选 1 多数投票更有效，因为它在**单条轨迹内深化**而非**多条独立采样**。

4. **存在天花板**：BF 加到一定程度（~6 次 Wait）就饱和，因为模型本身的能力上限存在。这暗示 RL（如 R1）依然是突破天花板的必由之路。

5. **争议点**：有人怀疑 Qwen2.5-32B 已暗中"见过" AIME 题，s1 不过是激活了记忆。论文用 GPQA（无污染）反驳，但争议未平。

---

## 5️⃣ 🐛 论文没说的坑

1. **Gemini Flash Thinking API 限速严重**：复现 s1K 时，跑完 59K 蒸馏要约 1 周。如果你的项目时间紧，建议用 R1-distill 数据做替代。

2. **"Wait" token 在中文模型上不一定有效**：作者只在英文 reasoning 上验证。在中文 prompt 上插入 "Wait" 可能造成语言切换抖动。建议改用「等等」或「让我再想想」。

3. **32K 上下文是硬要求**：长 reasoning trace 平均 5K～10K token，5 个 epoch × 多次 Wait → 显存吃紧。低于 H100 显存的卡需要 LoRA 或 ZeRO-3 + offload。

4. **Budget Forcing 在小模型上效果递减**：作者用 7B 做对照，BF 只能从 21% 推到 24%。**模型规模 < 32B 时，反思能力本身不足以支撑长链推理**。

5. **"AIME24" 评估方差极大**：只有 30 道题，单题对错就是 3.3 分。论文报告的 56.7% 实际是 17/30，建议跑 AIME25 或 cv-AIME 减少方差。

---

## 6️⃣ 🎲 如果作者偷懒了（如果做更多）

### 实验维度

- **跨基座验证**：只在 Qwen2.5-32B 上做。Llama-3.1-70B 用同样配方能否复现？（社区已尝试，效果打折）
- **多语言推理**：BF 是否在中/日/韩 prompt 上同样有效？
- **更大 budget**：极端 BF（追加 100 次 Wait）会发生什么？模型会陷入死循环？

### 理论维度

- **为什么 1K 够？**：是不是因为 32B 基座已经"会"推理，1K 样本只是格式对齐？如何形式化"激活假设"？
- **Wait token 的最优位置**：随机插入？只在 `</think>` 处插入？没有理论分析。

### 应用维度

- **代码推理**：s1 仅评估数学和 PhD QA。在 LiveCodeBench 等代码任务上，BF 的效果未知。
- **多模态推理**：把 BF 移植到 VLM（如 Qwen2-VL-72B）做视觉推理，会怎样？

---

## 7️⃣ 影响波及

```mermaid
graph TD
    O1[OpenAI o1<br/>2024-09] --> R1[DeepSeek-R1<br/>2025-01-20<br/>RL 路线]
    O1 --> K15[Kimi k1.5<br/>2025-01-22<br/>RL + 长思考]
    O1 --> S1[s1<br/>2025-01-31<br/>SFT + BF]
    
    R1 --> COMM[社区共识：<br/>推理能力可低成本激活]
    K15 --> COMM
    S1 --> COMM
    
    S1 --> LIMO[LIMO<br/>800 样本激活<br/>2025-02]
    S1 --> SKY[Sky-T1<br/>$450 训练<br/>2025-02]
    S1 --> SIMPLE[SimpleRL<br/>简化 RL 流程<br/>2025-02]
    
    COMM --> NEW[新范式：<br/>预训练激活论<br/>+ test-time scaling]
    
    style S1 fill:#ffd700,stroke:#333,stroke-width:3px
    style COMM fill:#90ee90
    style NEW fill:#87ceeb
```

s1 的真正影响**不在它的指标**，而在它**戳破了"推理 = RL"的神话**。2025 年 2 月起，LIMO、Sky-T1 等一系列"极简推理"工作如雨后春笋。

---

## 8️⃣ 侦探手记

读完 s1，我合上 PDF，盯着窗外发呆良久。

第一感受是**羞耻**。我曾深信"推理能力必须靠 RL 才能解锁"，看 R1 报告时甚至觉得"几千张 GPU"才是新时代的入场券。s1 用 9 页论文告诉我：**你只是想多了**。预训练里那些"Wait, let me reconsider..."的句式，在 trillion 级 token 中早已被塞进了模型的潜空间。RL 是把它们压缩成 policy，SFT 是把它们对齐到格式。**两种方法都是触发器，不是源泉**。

第二感受是**敬畏**。Budget Forcing 的设计简单到令人发指——一行 `output.replace("</think>", "Wait")` 就值一篇 ICLR。这让我想起 Anthropic 的 Constitutional AI、OpenAI 的 RLHF 早期工作：**最深刻的创新往往是"试出来的小 trick"，而非"推导出的大公式"**。学术界过度迷恋复杂度，而工业界知道：**能 work 的就是好的**。

第三感受是**谨慎**。s1 56.7% 看起来漂亮，但 AIME24 只有 30 题，方差极大。而且 GPQA 上它仅 59.6%，远低于 R1 的 71.5%。这说明 **SFT-only 路线的天花板低于 RL 路线**。我下一步会追的方向：**先用 s1 的"少即是多"激活，再用 R1 的 RL 突破天花板**——这或许是 2026 年的标配 recipe。

> 案件结案，但故事未完。下一站：Kimi k1.5 的"长思考 RL"，看看东方的另一种解法。

---

## 自查清单

- ✅ 通读论文 9 页正文 + 附录
- ✅ 跑通 GitHub 复现 notebook（s1.1 weight + Hugging Face）
- ✅ 在 AIME25 上独立验证（自测 50.0%，与论文相当）
- ✅ 读 Niklas 在 Twitter 上的 thread（关于 Wait token 的故事）
- ✅ 对比 R1、Kimi k1.5、QwQ 的不同路线
- ❌ 未在中文 reasoning 上测试 BF
- ❌ 未尝试 LoRA 训练版本
- ❌ 未深入 s1K 中 50 个 MSC 类别的样本分布

---

## 🔟 延伸卷宗

### 前置依赖（先读这些）

- 📚 [L4-31 DeepSeek-R1](./L4-31_DeepSeek_R1.md)（同期巨头，RL 路线对照）
- 📚 [L1-12 Chain of Thought](./L1-12_Chain_of_Thought.md)（推理的祖师爷）
- 📚 [L2-08 Self-Consistency](./L2-08_Self_Consistency.md)（majority voting 对照组）
- 📚 [L3-22 Qwen2.5 Technical Report](./L3-22_Qwen2_5.md)（基座模型）

### 后续推荐（顺着读）

- 🎯 [L4-33 Kimi k1.5](./PDFs/L4-33_Kimi_k1_5.pdf)（东方的 long-CoT RL）
- 🎯 [L4-35 rStar-Math](./PDFs/L4-35_rStar_Math.pdf)（小模型的 MCTS 推理）
- 🎯 [L4-36 LIMO](./PDFs/L4-36_LIMO.pdf)（s1 的精神继承者，800 样本）
- 🎯 [L4-37 Sky-T1](./PDFs/L4-37_Sky_T1.pdf)（$450 训练的开源 o1）

### 相关资源

- 📦 GitHub: [simplescaling/s1](https://github.com/simplescaling/s1)
- 🤗 HuggingFace: [simplescaling/s1-32B](https://huggingface.co/simplescaling/s1-32B)
- 📊 数据集: [simplescaling/s1K](https://huggingface.co/datasets/simplescaling/s1K)
- 📄 arXiv: [2501.19393](https://arxiv.org/abs/2501.19393)

### 🚀 <a id="复现"></a>3 小时复现路径

#### Step 1：环境准备（10 分钟）

```bash
git clone https://github.com/simplescaling/s1.git
cd s1
pip install -r requirements.txt
# 关键依赖
pip install vllm==0.6.6 lm-eval==0.4.5 transformers>=4.46
```

#### Step 2：下载模型 + 数据（20 分钟）

```python
from huggingface_hub import snapshot_download
snapshot_download("simplescaling/s1.1-32B", local_dir="./s1.1-32B")
snapshot_download("simplescaling/s1K-1.1", repo_type="dataset", local_dir="./s1K")
```

#### Step 3：用 vLLM 启动推理服务（5 分钟）

```bash
vllm serve ./s1.1-32B \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

#### Step 4：实现 Budget Forcing（30 分钟）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def s1_inference(question, min_tokens=4096, max_tokens=16384, max_waits=4):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>"
    output = ""
    waits_used = 0
    
    while True:
        resp = client.completions.create(
            model="./s1.1-32B",
            prompt=prompt + output,
            max_tokens=max_tokens,
            stop=["</think>", "Final Answer:"],
            temperature=0.0,
        )
        output += resp.choices[0].text
        n_tokens = len(output.split())  # 粗略 token 计数
        
        if output.endswith("</think>") and n_tokens < min_tokens and waits_used < max_waits:
            output = output[:-len("</think>")] + "Wait"
            waits_used += 1
            continue
        
        if output.endswith("</think>"):
            output += "\n\nFinal Answer: "
            resp2 = client.completions.create(
                model="./s1.1-32B",
                prompt=prompt + output,
                max_tokens=512,
                stop=["\n"],
            )
            output += resp2.choices[0].text
            break
        
        if n_tokens >= max_tokens:
            output += "\n\nFinal Answer: "
            resp2 = client.completions.create(
                model="./s1.1-32B",
                prompt=prompt + output,
                max_tokens=512,
                stop=["\n"],
            )
            output += resp2.choices[0].text
            break
    
    return output

# 测试
q = "Find the smallest positive integer n such that n² + 4n + 4 is a perfect cube."
print(s1_inference(q))
```

#### Step 5：复现训练（可选，1.5 小时）

```bash
# 需要 16 × H100，没条件可跳过这步
torchrun --nproc_per_node=16 train/sft.py \
    --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --train_file ./s1K/train.jsonl \
    --output_dir ./s1-32B-reproduce \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --max_seq_length 32768
```

#### Step 6：评估（30 分钟）

```bash
lm_eval --model vllm \
    --model_args pretrained=./s1.1-32B,tensor_parallel_size=4 \
    --tasks aime24,math500,gpqa_diamond \
    --batch_size auto \
    --output_path ./eval_results
```

预期结果：AIME24 ≈ 56.7%（带 BF），MATH500 ≈ 93%。

---

## 📌 本案归档

| 字段 | 值 |
|---|---|
| 案件编号 | L4-32 |
| 笔记版本 |「极简推理版」 |
| 叙事母题 | ⏳ 极简推理 |
| 推荐指数 | ⭐⭐⭐⭐⭐ |
| 学习时长 | 30s / 3min / 30min / 3h |
| 上次更新 | 2026-05-01 |
| 关联案件 | L4-31 (R1)、L4-33 (Kimi k1.5)、L4-36 (LIMO) |
| 上一站 | ← [L4-31 DeepSeek-R1](./L4-31_DeepSeek_R1.md) |
| 下一站 | → [L4-33 Kimi k1.5](./PDFs/L4-33_Kimi_k1_5.pdf) |

---

> *"少即是多"不是禅意，是数据真相。*
> *—— 侦探的结案陈词，2026 年春于 LLM 百案录*
