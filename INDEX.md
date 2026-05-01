# 📚《LLM 百案录》总索引

> **学习哲学**：每篇论文都是一桩"案件"——有动机、有凶器、有结案陈词。
> 你不是在"读论文"，你是在"破案"。

---

## 🗺️ 学习地图（建议路径）

```
                    ┌─ L1 基础地基（必经之路）
                    │
  入门          ────┼─ L2 训练与对齐（理解 ChatGPT 怎么炼成）
                    │
                    ├─ L3 架构与生态（MoE / Agent / RAG / PEFT）
                    │
  进阶          ────┴─ L4 前沿与应用（多模态 / 代码 / 医疗 / 安全）
```

**推荐阅读顺序**：
1. 🚇 **30秒入门版**：L1-01 → L1-02 → L1-03 → L1-11 → L1-12（5 篇看完，懂 LLM 大局）
2. 🚲 **1 周通读版**：L1 全部 + L2-01/05/14 + L3-01/07/15 + L3-21（理解工程实践）
3. 🚗 **1 月精读版**：所有 ⭐⭐⭐⭐⭐ 笔记 + 复现核心代码
4. 🚀 **3 月研究版**：全部读完，并跟进每篇的"延伸卷宗"

---

## 📑 L1 基础地基（Foundation）

> **主题**：Transformer、预训练范式、Prompt 工程的开山之作

| ID | 论文 | 推荐 | 叙事母题 | 一句话 |
|---|---|---|---|---|
| **L1-01** | [Attention Is All You Need](./notes/L1-01_Attention_Is_All_You_Need.md) | ⭐⭐⭐⭐⭐ | 🕵️ 创世悬案 | RNN 王朝的颠覆者，LLM 的入场券 |
| **L1-02** | [BERT](./notes/L1-02_BERT.md) | ⭐⭐⭐⭐⭐ | 📖 双向阅读 | NLP 的"读者"哲学 |
| **L1-03** | [GPT-1/2](./notes/L1-03_GPT1.md) | ⭐⭐⭐⭐⭐ | 📈 作家诞生 | 预测下一个词就能创造世界 |
| **L1-05** | [Neural Machine Translation](./notes/L1-05_Neural_Machine_Translation.md) | ⭐⭐⭐ | 🌐 早期翻译 | Seq2Seq 的雏形 |
| **L1-06** | [Seq2Seq](./notes/L1-06_Seq2Seq.md) | ⭐⭐⭐⭐ | 🔁 序列对序列 | Encoder-Decoder 范式 |
| **L1-07** | [Word2Vec](./notes/L1-07_Word2Vec.md) | ⭐⭐⭐⭐ | 🧬 语义向量 | 让"词"有了数学坐标 |
| **L1-08** | [Dropout](./notes/L1-08_Dropout.md) | ⭐⭐⭐⭐ | 🎲 随机抗过拟合 | 训练时让神经元"罢工" |
| **L1-09** | [LayerNorm](./notes/L1-09_LayerNorm.md) | ⭐⭐⭐⭐ | ⚖️ 稳态归一 | Transformer 的隐形地基 |
| **L1-10** | [Adam](./notes/L1-10_Adam.md) | ⭐⭐⭐⭐⭐ | 🎯 自适应优化 | 默认优化器之王 |
| **L1-11** | [GPT-3](./notes/L1-11_GPT3.md) | ⭐⭐⭐⭐⭐ | 💥 规模涌现 | 1750 亿参数引爆质变 |
| **L1-12** | [Chain of Thought](./notes/L1-12_Chain_of_Thought.md) | ⭐⭐⭐⭐⭐ | 🧠 思维链 | 让模型"出声思考" |
| **L1-13** | [Tree of Thoughts](./notes/L1-13_Tree_of_Thoughts.md) | ⭐⭐⭐⭐ | 🌳 思维树 | 让模型反悔与重选 |
| **L1-14** | [Language Models are Reasoners](./notes/L1-14_Language_Models_are_Reasoners.md) | ⭐⭐⭐⭐ | 🧮 推理者 | 涌现的推理能力 |
| **L1-15** | [Self-Consistency](./notes/L1-15_Self_Consistency.md) | ⭐⭐⭐⭐ | 🗳️ 多数派 | 多次采样投票 |
| **L1-17** | [LLaMA](./notes/L1-17_LLaMA.md) | ⭐⭐⭐⭐⭐ | 🦙 开源燎原 | 开源 LLM 的新基准 |

---

## 🛠️ L2 训练与对齐（Training & Alignment）

> **主题**：Scaling Laws、RLHF、DPO、长上下文、量化、归一化变体

### 🔥 Scaling 与训练理论

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L2-01** | [Scaling Laws](./notes/L2-01_Scaling_Laws.md) | ⭐⭐⭐⭐⭐ | 📐 规模法则 |
| **L2-03** | [Chinchilla](./notes/L2-03_Chinchilla.md) | ⭐⭐⭐⭐⭐ | 🦔 数据为王 |
| **L2-04** | [PaLM 2](./notes/L2-04_PaLM2.md) | ⭐⭐⭐⭐ | 🌍 反击 |

### ⚖️ 对齐与 RLHF 全家桶

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L2-05** | [InstructGPT / RLHF](./notes/L2-05_InstructGPT_RLHF.md) | ⭐⭐⭐⭐⭐ | 📚 三步走 |
| **L2-06** | [Fine-tuning Human Preferences](./notes/L2-06_Fine_tuning_Human_Preferences.md) | ⭐⭐⭐⭐⭐ | 📚 偏好调教 |
| **L2-07** | [Learning to Summarize](./notes/L2-07_Learning_to_Summarize.md) | ⭐⭐⭐ | 📖 摘要革命 |
| **L2-09** | [PPO](./notes/L2-09_PPO.md) | ⭐⭐⭐⭐⭐ | 🛡️ 信任域 |
| **L2-10** | [DeepRL from Human Preferences](./notes/L2-10_DeepRL_From_Human_Preferences.md) | ⭐⭐⭐⭐ | 🎮 游戏教练 |
| **L2-11** | [Reward Model Ensemble](./notes/L2-11_Reward_Model_Ensemble.md) | ⭐⭐⭐ | 🏆 集体智慧 |
| **L2-12** | [Constitutional AI](./notes/L2-12_Constitutional_AI.md) | ⭐⭐⭐⭐⭐ | 📜 法律先行 |
| **L2-13** | [RLHF Survey](./notes/L2-13_RLHF_Survey.md) | ⭐⭐⭐⭐ | 🗺️ 全景图 |
| **L2-14** | [DPO](./notes/L2-14_DPO.md) | ⭐⭐⭐⭐⭐ | ⚖️ 一步到位 |
| **L2-15** | [ORPO](./notes/L2-15_ORPO.md) | ⭐⭐⭐⭐ | 🔗 SFT + RLHF 合体 |
| **L2-16** | [DPO vs PPO](./notes/L2-16_DPO_vs_PPO.md) | ⭐⭐⭐⭐ | ⚖️ 路线之争 |
| **L2-17** | [RLAIF](./notes/L2-17_RLAIF.md) | ⭐⭐⭐ | 🤖 AI 民主 |
| **L2-18** | [Self-Rewarding LM](./notes/L2-18_Self_Rewarding_LM.md) | ⭐⭐⭐⭐ | 🔄 自我奖励 |

### 🧱 架构组件优化

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L2-19** | [RoPE](./notes/L2-19_RoPE.md) | ⭐⭐⭐⭐⭐ | 🔄 旋转位置 |
| **L2-20** | [ALiBi](./notes/L2-20_ALiBi.md) | ⭐⭐⭐⭐ | 📏 线性偏置 |
| **L2-21** | [FlashAttention](./notes/L2-21_FlashAttention.md) | ⭐⭐⭐⭐⭐ | ⚡ IO-aware |
| **L2-22** | [MQA](./notes/L2-22_MQA.md) | ⭐⭐⭐⭐ | 🗝️ 共享 KV |
| **L2-23** | [GLU Variants](./notes/L2-23_GLU_Variants.md) | ⭐⭐⭐⭐ | 🚪 门控变体 |
| **L2-24** | [RMSNorm](./notes/L2-24_RMSNorm.md) | ⭐⭐⭐⭐ | 🪶 极简归一 |
| **L2-25** | [Longformer](./notes/L2-25_Longformer.md) | ⭐⭐⭐⭐ | 🔭 稀疏注意 |
| **L2-26** | [GQA](./notes/L2-26_GQA.md) | ⭐⭐⭐⭐⭐ | 🗝️ 分组 KV |
| **L2-28** | [BFloat16](./notes/L2-28_BFloat16.md) | ⭐⭐⭐⭐ | 🔢 混合精度 |
| **L2-30** | [BigBird](./notes/L2-30_BigBird.md) | ⭐⭐⭐⭐ | 🐦 稀疏 + 全局 |

---

## 🌐 L3 架构与生态（Architecture & Ecosystem）

> **主题**：MoE、Agent、RAG、PEFT 四大支柱

### 🧩 Mixture of Experts

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L3-01** | [Mixtral](./notes/L3-01_Mixtral.md) | ⭐⭐⭐⭐⭐ | 🧩 专家混合 |
| **L3-02** | [ST-MoE](./notes/L3-02_ST_MoE.md) | ⭐⭐⭐⭐ | 🚦 稳定路由 |
| **L3-03** | [GShard](./notes/L3-03_GShard.md) | ⭐⭐⭐⭐ | 🌐 大规模分片 |
| **L3-04** | [Switch Transformer](./notes/L3-04_Switch_Transformer.md) | ⭐⭐⭐⭐⭐ | 🔀 简化门控 |
| **L3-06** | [Base Layers MoE](./notes/L3-06_BaseLayers_MoE.md) | ⭐⭐⭐⭐ | 🪜 平衡负载 |

### 🤖 Agent 生态

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L3-07** | [ReAct](./notes/L3-07_ReAct.md) | ⭐⭐⭐⭐⭐ | 🔁 思而后行 |
| **L3-09** | [Generative Agents](./notes/L3-09_Generative_Agents.md) | ⭐⭐⭐⭐⭐ | 🏘️ 虚拟小镇 |
| **L3-10** | [AutoGPT](./notes/L3-10_AutoGPT.md) | ⭐⭐⭐⭐ | 🤖 自主智能体 |
| **L3-11** | [HuggingGPT](./notes/L3-11_HuggingGPT.md) | ⭐⭐⭐⭐ | 🎮 调度中枢 |
| **L3-12** | [Visual Agent](./notes/L3-12_Visual_Agent.md) | ⭐⭐⭐⭐ | 🖼️ 视觉问答 |
| **L3-13** | [Toolformer](./notes/L3-13_Toolformer.md) | ⭐⭐⭐⭐⭐ | 🛠️ 工具学会 |
| **L3-13b** | [Tool Learning Code Llama](./notes/L3-13_Tool_Learning_CodeLlama.md) | ⭐⭐⭐ | 🛠️ 函数即工具 |

### 🔍 RAG 全家桶

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L3-14** | [WebGPT](./notes/L3-14_WebGPT.md) | ⭐⭐⭐⭐ | 🌐 浏览器 Agent |
| **L3-15** | [RAG](./notes/L3-15_RAG.md) | ⭐⭐⭐⭐⭐ | 📚 检索增强 |
| **L3-16** | [Atlas](./notes/L3-16_Atlas.md) | ⭐⭐⭐⭐ | 🗺️ 高效 RAG |
| **L3-17** | [Self-RAG](./notes/L3-17_Self_RAG.md) | ⭐⭐⭐⭐ | 🔍 自我反思 |
| **L3-18** | [Corrective RAG](./notes/L3-18_Corrective_RAG.md) | ⭐⭐⭐ | 🔧 纠错机制 |
| **L3-19** | [Query Augmentation](./notes/L3-19_RAG_Query_Augmentation.md) | ⭐⭐⭐ | 🔄 桥梁 |
| **L3-20** | [Knowledge Graph RAG](./notes/L3-20_Knowledge_Graph_RAG.md) | ⭐⭐⭐ | 🕸️ 结构化检索 |

### 🪶 PEFT（参数高效微调）

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L3-21** | [LoRA](./notes/L3-21_LoRA.md) | ⭐⭐⭐⭐⭐ | 🪶 低秩奇迹 |
| **L3-22** | [QLoRA](./notes/L3-22_QLoRA.md) | ⭐⭐⭐⭐⭐ | 🎯 极限压缩 |
| **L3-23** | [PEFT Survey](./notes/L3-23_PEFT.md) | ⭐⭐⭐⭐ | 📚 全景 |
| **L3-24** | [LoRA+](./notes/L3-24_LoRA_plus.md) | ⭐⭐⭐⭐ | ➕ 打破懒政 |
| **L3-25** | [DoRA](./notes/L3-25_DoRA.md) | ⭐⭐⭐⭐ | ⚖️ 解剖学 |
| **L3-26** | [AdapterHub](./notes/L3-26_AdapterHub.md) | ⭐⭐⭐⭐ | 🔌 USB 协议 |
| **L3-27** | [Prefix Tuning](./notes/L3-27_Prefix_Tuning.md) | ⭐⭐⭐⭐ | 🎪 主持人 |
| **L3-28** | [P-Tuning v2](./notes/L3-28_P_Tuning_v2.md) | ⭐⭐⭐⭐ | 🎯 深层引导 |
| **L3-29** | [IA³](./notes/L3-29_IA3.md) | ⭐⭐⭐ | ⚡ 课程表 |
| **L3-30** | [RLUT](./notes/L3-30_RLUT.md) | ⭐⭐⭐ | 📖 触类旁通 |

---

## 🚀 L4 前沿与应用（Frontier & Applications）

> **主题**：推理增强、新架构、长上下文、多模态、垂直领域、安全

### 🧠 推理增强

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L4-01** | [Let's Verify Step by Step](./notes/L4-01_Lets_Verify_Step_by_Step.md) | ⭐⭐⭐⭐⭐ | ✅ 过程验证 |
| **L4-02** | [Quiet-STaR](./notes/L4-02_Quiet_STaR.md) | ⭐⭐⭐ | ➕ 沉默思考 |
| **L4-03** | [MCTS for LLM](./notes/L4-03_MCTS_LLM.md) | ⭐⭐⭐ | 🌲 树搜索 |
| **L4-04** | [Process Reward Model](./notes/L4-04_Process_Reward_Model.md) | ⭐⭐⭐⭐ | ⭐ 过程打分 |
| **L4-05** | [STaR](./notes/L4-05_STaR.md) | ⭐⭐⭐⭐ | 🚀 自驱动 |

### 🌊 新架构（非 Transformer）

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L4-06** | [Mamba](./notes/L4-06_Mamba.md) | ⭐⭐⭐⭐ | 📖 状态空间回归 |
| **L4-07** | [Mamba 2](./notes/L4-07_Mamba2.md) | ⭐⭐⭐⭐ | 🌀 SSM 进化 |
| **L4-08** | [RetNet](./notes/L4-08_RetNet.md) | ⭐⭐⭐⭐ | 🔁 保留网络 |
| **L4-09** | [RWKV](./notes/L4-09_RWKV.md) | ⭐⭐⭐⭐ | 🪞 线性 RNN |
| **L4-10** | [Griffin](./notes/L4-10_Griffin.md) | ⭐⭐⭐⭐ | 🦅 混合架构 |

### 📜 长上下文

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L4-11** | [LM Infinite](./notes/L4-11_LM_Infinite.md) | ⭐⭐⭐ | 📚 无限 |
| **L4-12** | [PoSE](./notes/L4-12_PoSE.md) | ⭐⭐⭐⭐ | 🪄 位置外推 |
| **L4-13** | [Giraffe](./notes/L4-13_Giraffe.md) | ⭐⭐⭐ | 🦒 长颈伸展 |
| **L4-14** | [YaRN](./notes/L4-14_YaRN.md) | ⭐⭐⭐⭐ | 🧶 旋转伸缩 |

### 👁️ 多模态

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L4-15** | [GPT-4V](./notes/L4-15_GPT4V.md) | ⭐⭐⭐⭐⭐ | 👁️ 多模态王者 |
| **L4-16** | [LLaVA](./notes/L4-16_LLaVA.md) | ⭐⭐⭐⭐⭐ | 🤝 视觉指令 |
| **L4-17** | [MiniGPT-4](./notes/L4-17_MiniGPT4.md) | ⭐⭐⭐⭐ | 🔒 精简主义 |
| **L4-18** | [CogVLM](./notes/L4-18_CogVLM.md) | ⭐⭐⭐⭐ | 🔢 深度融合 |
| **L4-19** | [Fuyu-8B](./notes/L4-19_Fuyu8B.md) | ⭐⭐⭐⭐ | 📱 随身 AI |
| **L4-20** | [Kosmos-1](./notes/L4-20_Kosmos1.md) | ⭐⭐⭐⭐ | 🌍 原生感知 |

### 🛡️ 安全与可靠性

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L4-21** | [RLAP Safety](./notes/L4-21_RLAP_Safety.md) | ⭐⭐⭐⭐ | 🛡️ 预防为主 |
| **L4-22** | [Red Teaming LLM](./notes/L4-22_Red_Teaming_LLM.md) | ⭐⭐⭐⭐⭐ | 🐛 以攻促防 |
| **L4-23** | [LLM Fuzzing](./notes/L4-23_LLM_Fuzzing.md) | ⭐⭐⭐⭐ | 🧪 模糊测试 |
| **L4-25** | [Sycophancy](./notes/L4-25_Sycophancy.md) | ⭐⭐⭐⭐ | 🤥 谄媚问题 |

### 🏥 垂直领域

| ID | 论文 | 推荐 | 叙事母题 |
|---|---|---|---|
| **L4-24** | [LLMTime](./notes/L4-24_LLMTime.md) | ⭐⭐⭐⭐ | ⏰ 时间感知 |
| **L4-26** | [MedPaLM](./notes/L4-26_MedPaLM.md) | ⭐⭐⭐⭐ | 🏥 医学专才 |
| **L4-27** | [MedPaLM 2](./notes/L4-27_MedPaLM2.md) | ⭐⭐⭐⭐ | 🏥 专家级医学 |
| **L4-28** | [AlphaCode](./notes/L4-28_AlphaCode.md) | ⭐⭐⭐⭐⭐ | 💻 竞赛级代码 |
| **L4-29** | [CodeGen](./notes/L4-29_CodeGen.md) | ⭐⭐⭐⭐ | 🎓 代码工厂 |
| **L4-30** | [StarCoder](./notes/L4-30_StarCoder.md) | ⭐⭐⭐⭐ | ⭐ 开源代码 SOTA |

---

## 🎯 速查：按主题反向索引

### 🔥 入门必读（5 篇）
L1-01 Attention → L1-03 GPT → L1-12 CoT → L2-05 RLHF → L3-15 RAG

### 💰 工程师面试高频
L1-09 LayerNorm｜L1-10 Adam｜L2-19 RoPE｜L2-21 FlashAttention｜L2-26 GQA｜L3-21 LoRA｜L3-22 QLoRA｜L2-14 DPO

### 🏗️ 大模型搭建必修课
L1-01 Transformer → L2-19 RoPE → L2-21 FlashAttention → L2-22 MQA → L2-23 GLU → L2-24 RMSNorm → L1-17 LLaMA

### 🤖 想做 Agent 应用
L1-12 CoT → L1-13 ToT → L3-07 ReAct → L3-13 Toolformer → L3-15 RAG → L3-09 Generative Agents

### 🎨 想做 RAG 应用
L3-15 RAG → L3-17 Self-RAG → L3-18 Corrective RAG → L3-19 Query Augmentation → L3-20 KG-RAG

### 💸 想低成本微调
L3-21 LoRA → L3-22 QLoRA → L3-23 PEFT → L3-25 DoRA → L2-14 DPO

---

## 📊 本笔记集统计

- **论文数量**：97 篇（覆盖 2014-2024 主流 LLM 研究）
- **总字数**：约 60 万字
- **格式版本**：「案件体」（侦探叙事 + 多档学习路径）
- **维护者**：mixed
- **上次更新**：2026-04-30

---

## 🛠️ 笔记体例说明

### 五维雷达图（每个完整笔记都有）
```
创新性  ██████████ X/10  ← 核心思想的新颖程度
影响力  ██████████ X/10  ← 后续工作引用与产业落地
复杂度  ██████████ X/10  ← 数学/工程门槛
可复现  ██████████ X/10  ← 开源情况 + 算力门槛
争议度  ██████████ X/10  ← 学术争议程度
```

### 学习路径（按时间预算选）
- 🚇 **30 秒速览**：电梯里讲完一句话
- 🚲 **3 分钟通读**：搞懂 Why（动机）
- 🚗 **30 分钟精读**：搞懂 How（机制 + 公式 + 代码片段）
- 🚀 **3 小时复现**：动手写代码、跑实验

### 小节符号约定
- 0️⃣ 案件档案 — 时间/作者/凶器/陈词
- 1️⃣ 30 秒速览
- 2️⃣ 3 分钟通读（Why）
- 3️⃣ 30 分钟精读（How）
- 4️⃣ 物证清单（Results）+ 🔥 Hot Take
- 5️⃣ 🐛 论文没说的坑
- 6️⃣ 🎲 如果作者偷懒了（实验/理论缺口）
- 7️⃣ 影响波及（Impact graph）
- 8️⃣ 侦探手记（My Take）
- 🔟 延伸卷宗
- 🚀 3 小时复现路径
- 🎯 自查清单
- 📌 本案归档

---

> 💡 **使用技巧**：
> 1. 在 VSCode 里用 `Ctrl+Shift+F` 全文搜索关键词，比目录更快
> 2. 每篇笔记顶部都有 `🚇 30秒速览 ｜ 🚲 3分钟通读 ｜ 🚗 30分钟精读 ｜ 🚀 3小时复现` 锚点跳转
> 3. 不要追求"全部读完"——挑感兴趣的、按主题深度学习
