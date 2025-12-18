# Vol 20 LLM Agents   Tool Use
*Enriched by BITCOREOS | Phase 4 Batch 4*

---

### Hypothesis Hunting with Evolving Networks of Autonomous Scientific Agents
**Date:** 2025-10-13 | **Arxiv:** [2510.08619](https://hub.bitwiki.org/t/hypothesis-hunting-with-evolving-networks-of-autonomous-scientific-agents/16370)

#### Abstract
Large-scale scientific datasets -- spanning health biobanks, cell atlases, Earth reanalyses, and more -- create opportunities for exploratory discovery unconstrained by specific research questions. We term this process hypothesis hunting: the cumulative search for insight through sustained exploration across vast and complex hypothesis spaces. To support it, we introduce AScience, a framework modeling discovery as the interaction of agents, networks, and evaluation norms, and implement it as ASCollab, a distributed system of LLM-based research agents with heterogeneous behaviors. These agents self-organize into evolving networks, continually producing and peer-reviewing findings under shared standards of evaluation. Experiments show that such social dynamics enable the accumulation of expert-rated results along the diversity-quality-novelty frontier, including rediscoveries of established biomarkers, extensions of known pathways, and proposals of new therapeutic targets. While wet-lab validation remains indispensable, our experiments on cancer cohorts demonstrate that socially structured, agentic networks can sustain exploratory hypothesis hunting at scale.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### PEAR: Planner-Executor Agent Robustness Benchmark
**Date:** 2025-10-10 | **Arxiv:** [2510.07505](https://hub.bitwiki.org/t/pear-planner-executor-agent-robustness-benchmark/15883)

#### Abstract
Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Theory
* **Limits:** However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation.
* **Signal Tags:** #ai

---


### Iterated Agent for Symbolic Regression
**Date:** 2025-10-10 | **Arxiv:** [2510.08317](https://hub.bitwiki.org/t/iterated-agent-for-symbolic-regression/16067)

#### Abstract
Symbolic regression (SR), the automated discovery of mathematical expressions from data, is a cornerstone of scientific inquiry. However, it is often hindered by the combinatorial explosion of the search space and a tendency to overfit. Popular methods, rooted in genetic programming, explore this space syntactically, often yielding overly complex, uninterpretable models. This paper introduces IdeaSearchFitter, a framework that employs Large Language Models (LLMs) as semantic operators within an evolutionary search. By generating candidate expressions guided by natural-language rationales, our method biases discovery towards models that are not only accurate but also conceptually coherent and interpretable. We demonstrate IdeaSearchFitter's efficacy across diverse challenges: it achieves competitive, noise-robust performance on the Feynman Symbolic Regression Database (FSReD), outperforming several strong baselines; discovers mechanistically aligned models with good accuracy-complexity trade-offs on real-world data; and derives compact, physically-motivated parametrizations for Parton Distribution Functions in a frontier high-energy physics application. IdeaSearchFitter is a specialized module within our broader iterated agent framework, IdeaSearch, which is publicly available at https://www.ideasearch.cn/.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Application
* **Limits:** However, it is often hindered by the combinatorial explosion of the search space and a tendency to overfit.
* **Signal Tags:** #ai

---


### Inefficiencies of Meta Agents for Agent Design
**Date:** 2025-10-09 | **Arxiv:** [2510.06711](https://hub.bitwiki.org/t/inefficiencies-of-meta-agents-for-agent-design/15738)

#### Abstract
Recent works began to automate the design of agentic systems using meta-agents that propose and iteratively refine new agent architectures. In this paper, we examine three key challenges in a common class of meta-agents. First, we investigate how a meta-agent learns across iterations and find that simply expanding the context with all previous agents, as proposed by previous works, performs worse than ignoring prior designs entirely. We show that the performance improves with an evolutionary approach. Second, although the meta-agent designs multiple agents during training, it typically commits to a single agent at test time. We find that the designed agents have low behavioral diversity, limiting the potential for their complementary use. Third, we assess when automated design is economically viable. We find that only in a few cases--specifically, two datasets--the overall cost of designing and deploying the agents is lower than that of human-designed agents when deployed on over 15,000 examples. In contrast, the performance gains for other datasets do not justify the design cost, regardless of scale.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science
**Date:** 2025-10-07 | **Arxiv:** [2506.04405](https://hub.bitwiki.org/t/medagentgym-a-scalable-agentic-training-environment-for-code-centric-reasoning-in-biomedical-data-science/15258)

#### Abstract
We introduce MedAgentGym, a scalable and interactive training environment designed to enhance coding-based biomedical reasoning capabilities in large language model (LLM) agents. MedAgentGym comprises 72,413 task instances across 129 categories derived from 12 authentic real-world biomedical scenarios. Tasks are encapsulated within executable sandbox environments, each featuring detailed task specifications, interactive feedback mechanisms, verifiable ground truth annotations, and scalable training trajectory generation. Extensive benchmarking of 29 LLMs reveals substantial performance disparities in biomedical data science between commercial and open-source LLMs. Leveraging efficient multi-threaded and multi-turn trajectory sampling in MedAgentGym, Med-Copilot achieves performance gains of +43.02% and +45.28% from offline and online reinforcement learning, respectively, demonstrating MedAgentGym as an effective training ground while establishing itself as a cost-effective, privacy-preserving alternative competitive with proprietary LLMs (gpt-4o). By offering a unified execution environment with a comprehensive benchmark and accessible, extensible training resources, MedAgentGym delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical data science.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Risk-Sensitive Agent Compositions
**Date:** 2025-10-06 | **Arxiv:** [2506.04632](https://hub.bitwiki.org/t/risk-sensitive-agent-compositions/14608)

#### Abstract
From software development to robot control, modern agentic systems decompose complex objectives into a sequence of subtasks and choose a set of specialized AI agents to complete them. We formalize agentic workflows as directed acyclic graphs, called agent graphs, where edges represent AI agents and paths correspond to feasible compositions of agents. Real-world deployment requires selecting agent compositions that not only maximize task success but also minimize violations of safety, fairness, and privacy requirements which demands a careful analysis of the low-probability (tail) behaviors of compositions of agents. In this work, we consider risk minimization over the set of feasible agent compositions and seek to minimize the value-at-risk of the loss distribution of the agent composition where the loss quantifies violations of these requirements. We introduce an efficient algorithm which traverses the agent graph and finds a near-optimal composition of agents. It uses a dynamic programming approach to approximate the value-at-risk of agent compositions by exploiting a union bound. Furthermore, we prove that the approximation is near-optimal asymptotically for a broad class of practical loss functions. To evaluate our framework, we consider a suite of video game-like control benchmarks that require composing several agents trained with reinforcement learning and demonstrate our algorithm's effectiveness in approximating the value-at-risk and identifying the optimal agent composition.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### RANGER -- Repository-Level Agent for Graph-Enhanced Retrieval
**Date:** 2025-10-01 | **Arxiv:** [2509.25257](https://hub.bitwiki.org/t/ranger-repository-level-agent-for-graph-enhanced-retrieval/13437)

#### Abstract
General-purpose automated software engineering (ASE) includes tasks such as code completion, retrieval, repair, QA, and summarization. These tasks require a code retrieval system that can handle specific queries about code entities, or code entity queries (for example, locating a specific class or retrieving the dependencies of a function), as well as general queries without explicit code entities, or natural language queries (for example, describing a task and retrieving the corresponding code). We present RANGER, a repository-level code retrieval agent designed to address both query types, filling a gap in recent works that have focused primarily on code-entity queries. We first present a tool that constructs a comprehensive knowledge graph of the entire repository, capturing hierarchical and cross-file dependencies down to the variable level, and augments graph nodes with textual descriptions and embeddings to bridge the gap between code and natural language. RANGER then operates on this graph through a dual-stage retrieval pipeline. Entity-based queries are answered through fast Cypher lookups, while natural language queries are handled by MCTS-guided graph exploration. We evaluate RANGER across four diverse benchmarks that represent core ASE tasks including code search, question answering, cross-file dependency retrieval, and repository-level code completion. On CodeSearchNet and RepoQA it outperforms retrieval baselines that use embeddings from strong models such as Qwen3-8B. On RepoBench, it achieves superior cross-file dependency retrieval over baselines, and on CrossCodeEval, pairing RANGER with BM25 delivers the highest exact match rate in code completion compared to other RAG methods.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### World Modelling Improves Language Model Agents
**Date:** 2025-09-22 | **Arxiv:** [2506.02918](https://hub.bitwiki.org/t/world-modelling-improves-language-model-agents/10515)

#### Abstract
Tool use in stateful environments presents unique challenges for large language models (LLMs), where existing test-time compute strategies relying on repeated trials in the environment are impractical. We propose dynamics modelling (DyMo), a method that augments LLMs with a state prediction capability alongside function calling during post-training. This enables LLMs to predict the future states of their actions through an internal environment model. On the Berkeley Function Calling Leaderboard V2, DyMo improves success rates and significantly reduces hallucinations. We further integrate the internal environment model into self-verification sampling (SVS), and show that this substantially improves pass^k over number of trials k, and allows the model to refuse unreliable outputs. Together, DyMo and SVS greatly enhance the effectiveness and reliability of LLMs for tool use. We believe this work charts a path towards scalable planning RL methods for LLM inference without repeatedly querying the oracle environment.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Learning Representations in Video Game Agents with Supervised Contrastive Imitation Learning
**Date:** 2025-09-16 | **Arxiv:** [2509.11880](https://hub.bitwiki.org/t/learning-representations-in-video-game-agents-with-supervised-contrastive-imitation-learning/9648)

#### Abstract
This paper introduces a novel application of Supervised Contrastive Learning (SupCon) to Imitation Learning (IL), with a focus on learning more effective state representations for agents in video game environments. The goal is to obtain latent representations of the observations that capture better the action-relevant factors, thereby modeling better the cause-effect relationship from the observations that are mapped to the actions performed by the demonstrator, for example, the player jumps whenever an obstacle appears ahead. We propose an approach to integrate the SupCon loss with continuous output spaces, enabling SupCon to operate without constraints regarding the type of actions of the environment. Experiments on the 3D games Astro Bot and Returnal, and multiple 2D Atari games show improved representation quality, faster learning convergence, and better generalization compared to baseline models trained only with supervised action prediction loss functions.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Application
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### CAViAR: Critic-Augmented Video Agentic Reasoning
**Date:** 2025-09-10 | **Arxiv:** [2509.07680](https://hub.bitwiki.org/t/caviar-critic-augmented-video-agentic-reasoning/8731)

#### Abstract
Video understanding has seen significant progress in recent years, with models' performance on perception from short clips continuing to rise. Yet, multiple recent benchmarks, such as LVBench, Neptune, and ActivityNet-RTL, show performance wanes for tasks requiring complex reasoning on videos as queries grow more complex and videos grow longer. In this work, we ask: can existing perception capabilities be leveraged to successfully perform more complex video reasoning? In particular, we develop a large language model agent given access to video modules as subagents or tools. Rather than following a fixed procedure to solve queries as in previous work such as Visual Programming, ViperGPT, and MoReVQA, the agent uses the results of each call to a module to determine subsequent steps. Inspired by work in the textual reasoning domain, we introduce a critic to distinguish between instances of successful and unsuccessful sequences from the agent. We show that the combination of our agent and critic achieve strong performance on the previously-mentioned datasets.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Demo: Healthcare Agent Orchestrator (HAO) for Patient Summarization in Molecular Tumor Boards
**Date:** 2025-09-09 | **Arxiv:** [2509.06602](https://hub.bitwiki.org/t/demo-healthcare-agent-orchestrator-hao-for-patient-summarization-in-molecular-tumor-boards/8371)

#### Abstract
Molecular Tumor Boards (MTBs) are multidisciplinary forums where oncology specialists collaboratively assess complex patient cases to determine optimal treatment strategies. A central element of this process is the patient summary, typically compiled by a medical oncologist, radiation oncologist, or surgeon, or their trained medical assistant, who distills heterogeneous medical records into a concise narrative to facilitate discussion. This manual approach is often labor-intensive, subjective, and prone to omissions of critical information. To address these limitations, we introduce the Healthcare Agent Orchestrator (HAO), a Large Language Model (LLM)-driven AI agent that coordinates a multi-agent clinical workflow to generate accurate and comprehensive patient summaries for MTBs. Evaluating predicted patient summaries against ground truth presents additional challenges due to stylistic variation, ordering, synonym usage, and phrasing differences, which complicate the measurement of both succinctness and completeness. To overcome these evaluation hurdles, we propose TBFact, a ``model-as-a-judge'' framework designed to assess the comprehensiveness and succinctness of generated summaries. Using a benchmark dataset derived from de-identified tumor board discussions, we applied TBFact to evaluate our Patient History agent. Results show that the agent captured 94% of high-importance information (including partial entailments) and achieved a TBFact recall of 0.84 under strict entailment criteria. We further demonstrate that TBFact enables a data-free evaluation framework that institutions can deploy locally without sharing sensitive clinical data. Together, HAO and TBFact establish a robust foundation for delivering reliable and scalable support to MTBs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching
**Date:** 2025-09-08 | **Arxiv:** [2509.04752](https://hub.bitwiki.org/t/sepa-a-search-enhanced-predictive-agent-for-personalized-health-coaching/8124)

#### Abstract
This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group k-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff's $δ$=0.3, p=0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Reinforcement Learning for Aligning Large Language Models Agents with Interactive Environments: Quantifying and Mitigating Prompt Overfitting
**Date:** 2025-09-08 | **Arxiv:** [2410.19920](https://hub.bitwiki.org/t/reinforcement-learning-for-aligning-large-language-models-agents-with-interactive-environments-quantifying-and-mitigating-prompt-overfitting/8169)

#### Abstract
Reinforcement learning (RL) is a promising approach for aligning large language models (LLMs) knowledge with sequential decision-making tasks. However, few studies have thoroughly investigated the impact on LLM agents capabilities of fine-tuning them with RL in a specific environment. In this paper, we propose a novel framework to analyze the sensitivity of LLMs to prompt formulations following RL training in a textual environment. Our findings reveal that the performance of LLMs degrades when faced with prompt formulations different from those used during the RL training phase. Besides, we analyze the source of this sensitivity by examining the model's internal representations and salient tokens. Finally, we propose to use a contrastive loss to mitigate this sensitivity and improve the robustness and generalization capabilities of LLMs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Infrastructure
* **Limits:** However, few studies have thoroughly investigated the impact on LLM agents capabilities of fine-tuning them with RL in a specific environment.
* **Signal Tags:** #ai

---


### Throttling Web Agents Using Reasoning Gates
**Date:** 2025-09-03 | **Arxiv:** [2509.01619](https://hub.bitwiki.org/t/throttling-web-agents-using-reasoning-gates/7362)

#### Abstract
AI web agents use Internet resources at far greater speed, scale, and complexity -- changing how users and services interact. Deployed maliciously or erroneously, these agents could overload content providers. At the same time, web agents can bypass CAPTCHAs and other defenses by mimicking user behavior or flood authentication systems with fake accounts. Yet providers must protect their services and content from denial-of-service attacks and scraping by web agents. In this paper, we design a framework that imposes tunable costs on agents before providing access to resources; we call this Web Agent Throttling. We start by formalizing Throttling Gates as challenges issued to an agent that are asymmetric, scalable, robust, and compatible with any agent. Focusing on a common component -- the language model -- we require the agent to solve reasoning puzzles, thereby incurring excessive token-generation costs. However, we find that using existing puzzles, e.g., coding or math, as throttling gates fails to satisfy our properties. To address this, we introduce rebus-based Reasoning Gates, synthetic text puzzles that require multi-hop reasoning over world knowledge (thereby throttling an agent's model). We design a scalable generation and verification protocol for such reasoning gates. Our framework achieves computational asymmetry, i.e., the response-generation cost is 9.2x higher than the generation cost for SOTA models. We further deploy reasoning gates on a custom website and Model Context Protocol (MCP) servers and evaluate with real-world web agents. Finally, we discuss the limitations and environmental impact of real-world deployment of our framework.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Infrastructure
* **Limits:** However, we find that using existing puzzles, e.
* **Signal Tags:** #ai

---


### Towards Agent-based Test Support Systems: An Unsupervised Environment Design Approach
**Date:** 2025-08-21 | **Arxiv:** [2508.14135](https://hub.bitwiki.org/t/towards-agent-based-test-support-systems-an-unsupervised-environment-design-approach/5081)

#### Abstract
Modal testing plays a critical role in structural analysis by providing essential insights into dynamic behaviour across a wide range of engineering industries. In practice, designing an effective modal test campaign involves complex experimental planning, comprising a series of interdependent decisions that significantly influence the final test outcome. Traditional approaches to test design are typically static-focusing only on global tests without accounting for evolving test campaign parameters or the impact of such changes on previously established decisions, such as sensor configurations, which have been found to significantly influence test outcomes. These rigid methodologies often compromise test accuracy and adaptability. To address these limitations, this study introduces an agent-based decision support framework for adaptive sensor placement across dynamically changing modal test environments. The framework formulates the problem using an underspecified partially observable Markov decision process, enabling the training of a generalist reinforcement learning agent through a dual-curriculum learning strategy. A detailed case study on a steel cantilever structure demonstrates the efficacy of the proposed method in optimising sensor locations across frequency segments, validating its robustness and real-world applicability in experimental settings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents
**Date:** 2025-08-12 | **Arxiv:** [2505.19997](https://hub.bitwiki.org/t/embracing-imperfection-simulating-students-with-diverse-cognitive-levels-using-llm-based-agents/3100)

#### Abstract
Large language models (LLMs) are revolutionizing education, with LLM-based agents playing a key role in simulating student behavior. A major challenge in student simulation is modeling the diverse learning patterns of students at various cognitive levels. However, current LLMs, typically trained as ``helpful assistants'', target at generating perfect responses. As a result, they struggle to simulate students with diverse cognitive abilities, as they often produce overly advanced answers, missing the natural imperfections that characterize student learning and resulting in unrealistic simulations. To address this issue, we propose a training-free framework for student simulation. We begin by constructing a cognitive prototype for each student using a knowledge graph, which captures their understanding of concepts from past learning records. This prototype is then mapped to new tasks to predict student performance. Next, we simulate student solutions based on these predictions and iteratively refine them using a beam search method to better replicate realistic mistakes. To validate our approach, we construct the \texttt{Student\_100} dataset, consisting of $100$ students working on Python programming and $5,000$ learning records. Experimental results show that our method consistently outperforms baseline models, achieving $100\%$ improvement in simulation accuracy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Not Mentioned
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Infrastructure
* **Limits:** However, current LLMs, typically trained as ``helpful assistants'', target at generating perfect responses.
* **Signal Tags:** #ai

---
