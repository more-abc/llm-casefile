# 🕵️ LLM 百案录

> **把每一篇大语言模型核心论文都当作一桩"案件"——
> 有动机、有凶器、有关键证据、有结案陈词。**

<p align="center">
  <img src="https://img.shields.io/badge/papers-100%2B-42b983?style=flat-square" alt="papers" />
  <img src="https://img.shields.io/badge/format-v5%20%E6%A1%88%E4%BB%B6%E4%BD%93-blueviolet?style=flat-square" alt="format" />
  <img src="https://img.shields.io/badge/language-%E4%B8%AD%E6%96%87-red?style=flat-square" alt="lang" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license" />
  <img src="https://img.shields.io/badge/last%20update-2026--04--30-lightgrey?style=flat-square" alt="update" />
</p>

<p align="center">
  <a href="#-30-秒快速开始">🚀 快速开始</a> ·
  <a href="./INDEX.md">📑 完整目录</a> ·
  <a href="#-阅读路径">🛣️ 阅读路径</a> ·
  <a href="#-本地预览--部署">🛠️ 本地预览</a> ·
  <a href="./CONTRIBUTING.md">🤝 参与贡献</a>
</p>

---

## ✨ 这是什么？

**《LLM 百案录》** 是一套系统化的大语言模型核心论文中文笔记集，覆盖 **2014—2024** 年的 **100+ 篇** 关键工作。

每一篇笔记都遵循统一的 **「案件体」** 格式——把论文当案件来"破"：

- 0️⃣ **案件档案**：时间 / 作者 / 凶器 / 结案陈词
- 1️⃣ **30 秒速览**：电梯里讲完一句话
- 2️⃣ **3 分钟通读**：搞懂 *Why*（动机）
- 3️⃣ **30 分钟精读**：搞懂 *How*（机制 + 公式 + 代码片段）
- 4️⃣ **物证清单 & 🔥 Hot Take**
- 5️⃣ **🐛 论文没说的坑**
- 6️⃣ **🎲 如果作者偷懒了**（实验/理论缺口）
- 7️⃣ **影响波及**（mermaid 影响图）
- 8️⃣ **侦探手记**（My Take）
- 🚀 **3 小时复现路径**（部分论文）

> 你不是在读论文，你是在**破案**。

---

## 🚇 30 秒快速开始

| 你是…… | 推荐 5 篇入门 |
|---|---|
| 🌱 **完全新手** | [Attention](./L1-01_Attention_Is_All_You_Need.md) → [BERT](./L1-02_BERT.md) → [GPT-1](./L1-03_GPT1.md) → [GPT-3](./L1-11_GPT3.md) → [CoT](./L1-12_Chain_of_Thought.md) |
| 💼 **算法面试** | [LayerNorm](./L1-09_LayerNorm.md) · [Adam](./L1-10_Adam.md) · [RoPE](./L2-19_RoPE.md) · [FlashAttention](./L2-21_FlashAttention.md) · [LoRA](./L3-21_LoRA.md) |
| 🤖 **做 Agent** | [CoT](./L1-12_Chain_of_Thought.md) → [ReAct](./L3-07_ReAct.md) → [Toolformer](./L3-13_Toolformer.md) → [Generative Agents](./L3-09_Generative_Agents.md) → [Visual Agent](./L3-12_Visual_Agent.md) |
| 🔍 **做 RAG** | [RAG](./L3-15_RAG.md) → [Self-RAG](./L3-17_Self_RAG.md) → [Corrective RAG](./L3-18_Corrective_RAG.md) → [Query Aug](./L3-19_RAG_Query_Augmentation.md) → [KG-RAG](./L3-20_Knowledge_Graph_RAG.md) |
| 💸 **想低成本微调** | [LoRA](./L3-21_LoRA.md) → [QLoRA](./L3-22_QLoRA.md) → [DoRA](./L3-25_DoRA.md) → [DPO](./L2-14_DPO.md) → [ORPO](./L2-15_ORPO.md) |
| 🏗️ **想搭大模型** | [Transformer](./L1-01_Attention_Is_All_You_Need.md) → [RoPE](./L2-19_RoPE.md) → [FlashAttn](./L2-21_FlashAttention.md) → [GQA](./L2-26_GQA.md) → [LLaMA](./L1-17_LLaMA.md) |

---

## 📚 论文地图（4 大层级 · 100+ 篇）

```
┌─ L1 基础地基 ─────────── Transformer / BERT / GPT / Word2Vec / CoT …
│
├─ L2 训练与对齐 ─────── Scaling Laws / RLHF / DPO / RoPE / FlashAttn …
│
├─ L3 架构与生态 ─────── MoE / Agent / RAG / PEFT 四大支柱
│
└─ L4 前沿与应用 ─────── 推理增强 / 新架构 / 长上下文 / 多模态 / 安全 …
```

📑 **完整章节索引**：[INDEX.md](./INDEX.md)

### 高亮章节速链

#### L1 基础地基（15 篇）
[Attention](./L1-01_Attention_Is_All_You_Need.md) · [BERT](./L1-02_BERT.md) · [GPT-1/2](./L1-03_GPT1.md) · [Seq2Seq](./L1-06_Seq2Seq.md) · [Word2Vec](./L1-07_Word2Vec.md) · [Dropout](./L1-08_Dropout.md) · [LayerNorm](./L1-09_LayerNorm.md) · [Adam](./L1-10_Adam.md) · [GPT-3](./L1-11_GPT3.md) · [CoT](./L1-12_Chain_of_Thought.md) · [ToT](./L1-13_Tree_of_Thoughts.md) · [Self-Consistency](./L1-15_Self_Consistency.md) · [LLaMA](./L1-17_LLaMA.md)

#### L2 训练与对齐（28 篇）
[Scaling Laws](./L2-01_Scaling_Laws.md) · [Chinchilla](./L2-03_Chinchilla.md) · [InstructGPT](./L2-05_InstructGPT_RLHF.md) · [PPO](./L2-09_PPO.md) · [Constitutional AI](./L2-12_Constitutional_AI.md) · [DPO](./L2-14_DPO.md) · [ORPO](./L2-15_ORPO.md) · [Self-Rewarding](./L2-18_Self_Rewarding_LM.md) · [RoPE](./L2-19_RoPE.md) · [FlashAttention](./L2-21_FlashAttention.md) · [MQA](./L2-22_MQA.md) · [GLU Variants](./L2-23_GLU_Variants.md) · [RMSNorm](./L2-24_RMSNorm.md) · [GQA](./L2-26_GQA.md) · [BFloat16](./L2-28_BFloat16.md) · …

#### L3 架构与生态（30 篇）
[Mixtral](./L3-01_Mixtral.md) · [Switch Transformer](./L3-04_Switch_Transformer.md) · [ReAct](./L3-07_ReAct.md) · [Generative Agents](./L3-09_Generative_Agents.md) · [AutoGPT](./L3-10_AutoGPT.md) · [Visual Agent](./L3-12_Visual_Agent.md) · [Toolformer](./L3-13_Toolformer.md) · [WebGPT](./L3-14_WebGPT.md) · [RAG](./L3-15_RAG.md) · [Self-RAG](./L3-17_Self_RAG.md) · [LoRA](./L3-21_LoRA.md) · [QLoRA](./L3-22_QLoRA.md) · [DoRA](./L3-25_DoRA.md) · [Prefix Tuning](./L3-27_Prefix_Tuning.md) · …

#### L4 前沿与应用（30 篇）
[Step-by-Step Verify](./L4-01_Lets_Verify_Step_by_Step.md) · [STaR](./L4-05_STaR.md) · [Mamba](./L4-06_Mamba.md) · [Mamba 2](./L4-07_Mamba2.md) · [RetNet](./L4-08_RetNet.md) · [RWKV](./L4-09_RWKV.md) · [YaRN](./L4-14_YaRN.md) · [GPT-4V](./L4-15_GPT4V.md) · [LLaVA](./L4-16_LLaVA.md) · [CogVLM](./L4-18_CogVLM.md) · [Red Teaming](./L4-22_Red_Teaming_LLM.md) · [Sycophancy](./L4-25_Sycophancy.md) · [MedPaLM 2](./L4-27_MedPaLM2.md) · [AlphaCode](./L4-28_AlphaCode.md) · [StarCoder](./L4-30_StarCoder.md) · …

---

## 🛣️ 阅读路径

| 时间预算 | 路径 | 适合 |
|---|---|---|
| ⏱️ **30 秒** | 看 5 篇⭐⭐⭐⭐⭐入门必读 | 朋友圈装个内行 |
| 📅 **1 周通读** | L1 全部 + L2-01/05/14 + L3-01/07/15/21 | 面试 / 转岗 |
| 📅 **1 个月精读** | 所有 ⭐⭐⭐⭐⭐ + 复现核心代码 | 工程师上岗 |
| 📅 **3 个月研究** | 全部读完 + 跟进每篇延伸卷宗 | 做 LLM 研究 |

---

## 🛠️ 本地预览 / 部署

### 方式 1：直接打开（无需任何环境）
所有笔记都是纯 Markdown，**任何 Markdown 阅读器** 都能直接读：
- VSCode（推荐：装 `Markdown Preview Enhanced` 插件）
- Typora、Obsidian、Marktext、Logseq …

### 方式 2：本地起 Docsify 站点
```bash
# 任选其一
npx docsify-cli serve .
# 或
python -m http.server 3000
# 或
npm i -g docsify-cli && docsify serve .
```
然后打开 <http://localhost:3000> 即可。

### 方式 3：部署到 GitHub Pages（一键自动）
1. Fork 本仓库到自己账号
2. 进入 **Settings → Pages**
3. **Source**：选择 `Deploy from a branch`
4. **Branch**：`main` / `(root)`，保存
5. 几十秒后访问 `https://<your-username>.github.io/<repo-name>/`

> 仓库已包含 `index.html`、`.nojekyll`、`_sidebar.md`、`_navbar.md`、`_coverpage.md`、
> 以及 `.github/workflows/pages.yml`——可直接 Pages，零配置。

---

## 🧱 仓库结构

```
.
├── index.html              # Docsify 入口（GitHub Pages 静态站点）
├── README.md               # GitHub 首页 / Docsify 主页（你正在看）
├── INDEX.md                # 完整论文目录（详细分类）
├── _sidebar.md             # Docsify 左侧导航
├── _navbar.md              # Docsify 顶部导航
├── _coverpage.md           # Docsify 封面页
├── .nojekyll               # 关闭 GitHub Pages 的 Jekyll
├── LICENSE                 # MIT
├── CONTRIBUTING.md         # 贡献指南
├── .gitignore
├── .github/workflows/
│   └── pages.yml           # 自动 deploy 到 GitHub Pages
│
├── L1-01_Attention_Is_All_You_Need.md
├── L1-02_BERT.md
├── …
├── L4-30_StarCoder.md
└── (100 篇 案件体笔记)
```

> 📌 **设计选择**：所有笔记保持**扁平结构**（不分子目录），是为了让所有内部链接 `./Lx-yy_xxx.md` 在 GitHub 网页 / Docsify / 本地阅读器中都能直接跳转——零迁移成本。

---

## 🎨 笔记体例速查

### 五维雷达图（每个完整笔记都有）
```
创新性  ██████████ X/10  ← 核心思想的新颖程度
影响力  ██████████ X/10  ← 后续工作引用与产业落地
复杂度  ██████████ X/10  ← 数学/工程门槛
可复现  ██████████ X/10  ← 开源情况 + 算力门槛
争议度  ██████████ X/10  ← 学术争议程度
```

### 学习路径锚点
每篇笔记顶部都有 4 档跳转：
> 🚇 [30秒速览](#速览) ｜ 🚲 [3分钟通读](#通读) ｜ 🚗 [30分钟精读](#精读) ｜ 🚀 [3小时复现](#复现)

---

## 🤝 参与贡献

PR 永远欢迎！可贡献的方向：

- 🆕 新增论文笔记（请遵循 **案件体** 格式，参考任意已有笔记作为模板）
- 🐛 修正事实错误 / 拼写 / 链接失效
- 🌐 翻译成英文 / 其他语言
- 🎨 完善 Docsify 主题、添加新插件
- 📝 补充"3 小时复现"代码片段

详情见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

### 友情致谢
本笔记集站在巨人的肩膀上——所有原始论文作者贡献了真正的智慧，本仓库只是把它们整理成更易消化的"案件卷宗"。

---

## 📜 License

[MIT](./LICENSE) · 笔记内容采用知识共享理念：自由复制、修改、传播，唯一要求是保留来源。

> 💡 **附注**：所有原始论文版权归原作者所有；本仓库仅为学习笔记与二次创作。

---

## 🌟 Star History

如果这个仓库帮到你了，给个 ⭐ 是对作者最大的鼓励。

<p align="center">
  <i>📚 把论文当案件来读，你会发现整个 LLM 时代是一部连续的侦探小说。</i>
</p>
