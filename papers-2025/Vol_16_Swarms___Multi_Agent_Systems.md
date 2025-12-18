# Vol 16 Swarms   Multi Agent Systems
*Enriched by BITCOREOS | Phase 4 Batch 4*

---

### Asynchronous Risk-Aware Multi-Agent Packet Routing for Ultra-Dense LEO Satellite Networks
**Date:** 2025-11-03 | **Arxiv:** [2510.27506](https://arxiv.org/abs/2510.27506)

#### Abstract
The rise of ultra-dense LEO constellations creates a complex and asynchronous network environment, driven by their massive scale, dynamic topologies, and significant delays. This unique complexity demands an adaptive packet routing algorithm that is asynchronous, risk-aware, and capable of balancing diverse and often conflicting QoS objectives in a decentralized manner. However, existing methods fail to address this need, as they typically rely on impractical synchronous decision-making and/or risk-oblivious approaches. To tackle this gap, we introduce PRIMAL, an event-driven multi-agent routing framework designed specifically to allow each satellite to act independently on its own event-driven timeline, while managing the risk of worst-case performance degradation via a principled primal-dual approach. This is achieved by enabling agents to learn the full cost distribution of the targeted QoS objectives and constrain tail-end risks. Extensive simulations on a LEO constellation with 1584 satellites validate its superiority in effectively optimizing latency and balancing load. Compared to a recent risk-oblivious baseline, it reduces queuing delay by over 70%, and achieves a nearly 12 ms end-to-end delay reduction in loaded scenarios. This is accomplished by resolving the core conflict between naive shortest-path finding and congestion avoidance, highlighting such autonomous risk-awareness as a key to robust routing.

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
* **Limits:** However, existing methods fail to address this need, as they typically rely on impractical synchronous decision-making and/or risk-oblivious approaches.
* **Signal Tags:** #ai

---


### Network-Constrained Policy Optimization for Adaptive Multi-agent Vehicle Routing
**Date:** 2025-10-31 | **Arxiv:** [2510.26089](https://arxiv.org/abs/2510.26089)

#### Abstract
Traffic congestion in urban road networks leads to longer trip times and higher emissions, especially during peak periods. While the Shortest Path First (SPF) algorithm is optimal for a single vehicle in a static network, it performs poorly in dynamic, multi-vehicle settings, often worsening congestion by routing all vehicles along identical paths. We address dynamic vehicle routing through a multi-agent reinforcement learning (MARL) framework for coordinated, network-aware fleet navigation. We first propose Adaptive Navigation (AN), a decentralized MARL model where each intersection agent provides routing guidance based on (i) local traffic and (ii) neighborhood state modeled using Graph Attention Networks (GAT). To improve scalability in large networks, we further propose Hierarchical Hub-based Adaptive Navigation (HHAN), an extension of AN that assigns agents only to key intersections (hubs). Vehicles are routed hub-to-hub under agent control, while SPF handles micro-routing within each hub region. For hub coordination, HHAN adopts centralized training with decentralized execution (CTDE) under the Attentive Q-Mixing (A-QMIX) framework, which aggregates asynchronous vehicle decisions via attention. Hub agents use flow-aware state features that combine local congestion and predictive dynamics for proactive routing. Experiments on synthetic grids and real urban maps (Toronto, Manhattan) show that AN reduces average travel time versus SPF and learning baselines, maintaining 100% routing success. HHAN scales to networks with hundreds of intersections, achieving up to 15.9% improvement under heavy traffic. These findings highlight the potential of network-constrained MARL for scalable, coordinated, and congestion-aware routing in intelligent transportation systems.

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


### MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks
**Date:** 2025-10-31 | **Arxiv:** [2505.12371](https://arxiv.org/abs/2505.12371)

#### Abstract
The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at https://medagentboard.netlify.app/.

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
* **Limits:** However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood.
* **Signal Tags:** #ai

---


### Fortytwo: Swarm Inference with Peer-Ranked Consensus
**Date:** 2025-10-30 | **Arxiv:** [2510.24801](https://arxiv.org/abs/2510.24801)

#### Abstract
As centralized AI hits compute ceilings and diminishing returns from ever-larger training runs, meeting demand requires an inference layer that scales horizontally in both capacity and capability. We present Fortytwo, a novel protocol that leverages swarm intelligence principles and distributed pairwise ranking consensus to achieve superior performance in AI inference. Our approach reimagines collaboration among AI nodes using swarm inference: a peer-ranked, reputation-weighted consensus across heterogeneous models that surfaces the highest-quality responses. Using pairwise ranking with a custom Bradley-Terry-style aggregation model, we demonstrate that swarm inference substantially outperforms majority voting, achieving 85.90% on GPQA Diamond versus 68.69% for majority voting with the same model set - an improvement of +17.21 percentage points (approximately +25.1% relative). The protocol incorporates on-chain reputation so node influence adapts to demonstrated accuracy over time, yielding a meritocratic consensus that filters low-quality or malicious participants. To resist Sybil attacks, Fortytwo employs proof-of-capability in its consensus: nodes must successfully complete calibration/test requests and stake reputation to enter ranking rounds, making multi-identity attacks economically unattractive while preserving openness. Across six challenging benchmarks, including GPQA Diamond, LiveCodeBench, and AIME, our evaluation indicates higher accuracy and strong resilience to adversarial and noisy free-form prompting (e.g., prompt-injection degradation of only 0.12% versus 6.20% for a monolithic single-model baseline), while retaining practical deployability. Together, these results establish a foundation for decentralized AI systems - democratizing access to high-quality inference through collective intelligence without sacrificing reliability or security.

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
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective
**Date:** 2025-10-27 | **Arxiv:** [2505.20922](https://arxiv.org/abs/2505.20922)

#### Abstract
World models have recently attracted growing interest in Multi-Agent Reinforcement Learning (MARL) due to their ability to improve sample efficiency for policy learning. However, accurately modeling environments in MARL is challenging due to the exponentially large joint action space and highly uncertain dynamics inherent in multi-agent systems. To address this, we reduce modeling complexity by shifting from jointly modeling the entire state-action transition dynamics to focusing on the state space alone at each timestep through sequential agent modeling. Specifically, our approach enables the model to progressively resolve uncertainty while capturing the structured dependencies among agents, providing a more accurate representation of how agents influence the state. Interestingly, this sequential revelation of agents' actions in a multi-agent system aligns with the reverse process in diffusion models--a class of powerful generative models known for their expressiveness and training stability compared to autoregressive or latent variable models. Leveraging this insight, we develop a flexible and robust world model for MARL using diffusion models. Our method, Diffusion-Inspired Multi-Agent world model (DIMA), achieves state-of-the-art performance across multiple multi-agent control benchmarks, significantly outperforming prior world models in terms of final return and sample efficiency, including MAMuJoCo and Bi-DexHands. DIMA establishes a new paradigm for constructing multi-agent world models, advancing the frontier of MARL research. Codes are open-sourced at https://github.com/breez3young/DIMA.

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
* **Limits:** However, accurately modeling environments in MARL is challenging due to the exponentially large joint action space and highly uncertain dynamics inherent in multi-agent systems.
* **Signal Tags:** #ai

---


### Mitigating Manipulation and Enhancing Persuasion: A Reflective Multi-Agent Approach for Legal Argument Generation
**Date:** 2025-10-27 | **Arxiv:** [2506.02992](https://arxiv.org/abs/2506.02992)

#### Abstract
Large Language Models (LLMs) are increasingly explored for legal argument generation, yet they pose significant risks of manipulation through hallucination and ungrounded persuasion, and often fail to utilize provided factual bases effectively or abstain when arguments are untenable. This paper introduces a novel reflective multi-agent method designed to address these challenges in the context of legally compliant persuasion. Our approach employs specialized agents (factor analyst and argument polisher) in an iterative refinement process to generate 3-ply legal arguments (plaintiff, defendant, rebuttal). We evaluate reflective multi-agent against single-agent, enhanced-prompt single-agent, and non-reflective multi-agent baselines using four diverse LLMs (GPT-4o, GPT-4o-mini, Llama-4-Maverick-17b-128e, Llama-4-Scout-17b-16e) across three legal scenarios: "arguable", "mismatched", and "non-arguable". Results demonstrate that the reflective multi-agent approach excels at successful abstention by preventing generation when arguments cannot be grounded, improves hallucination accuracy by reducing fabricated and misattributed factors and enhances factor utilization recall by better using the provided case facts. These findings suggest that structured reflection within a multi-agent framework offers a robust method for fostering ethical persuasion and mitigating manipulation in LLM-based legal argumentation systems.

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


### Breaking and Fixing Defenses Against Control-Flow Hijacking in Multi-Agent Systems
**Date:** 2025-10-21 | **Arxiv:** [2510.17276](https://arxiv.org/abs/2510.17276)

#### Abstract
Control-flow hijacking attacks manipulate orchestration mechanisms in multi-agent systems into performing unsafe actions that compromise the system and exfiltrate sensitive information. Recently proposed defenses, such as LlamaFirewall, rely on alignment checks of inter-agent communications to ensure that all agent invocations are "related to" and "likely to further" the original objective.   We start by demonstrating control-flow hijacking attacks that evade these defenses even if alignment checks are performed by advanced LLMs. We argue that the safety and functionality objectives of multi-agent systems fundamentally conflict with each other. This conflict is exacerbated by the brittle definitions of "alignment" and the checkers' incomplete visibility into the execution context.   We then propose, implement, and evaluate ControlValve, a new defense inspired by the principles of control-flow integrity and least privilege. ControlValve (1) generates permitted control-flow graphs for multi-agent systems, and (2) enforces that all executions comply with these graphs, along with contextual rules (generated in a zero-shot manner) for each agent invocation.

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
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration
**Date:** 2025-10-21 | **Arxiv:** [2510.16645](https://arxiv.org/abs/2510.16645)

#### Abstract
Large Language Models (LLMs) demonstrate strong performance but often lack interpretable reasoning. This paper introduces the Multi-Agent Collaboration Framework for Diverse Thinking Modes (DiMo), which enhances both performance and interpretability by simulating a structured debate among four specialized LLM agents. Each agent embodies a distinct reasoning paradigm, allowing the framework to collaboratively explore diverse cognitive approaches. Through iterative debate, agents challenge and refine initial responses, yielding more robust conclusions and an explicit, auditable reasoning chain. Across six benchmarks and under a unified open-source setup, DiMo improves accuracy over widely used single-model and debate baselines, with the largest gains on math. We position DiMo as a semantics-aware, Web-native multi-agent framework: it models human-machine intelligence with LLM agents that produce semantically typed, URL-annotated evidence chains for explanations and user-friendly interactions. Although our experiments use standard reasoning benchmarks, the framework is designed to be instantiated over Web corpora and knowledge graphs, combining retrieval-augmented reasoning with structured justifications that downstream systems can inspect and reuse.

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


### ARCO-BO: Adaptive Resource-aware COllaborative Bayesian Optimization for Heterogeneous Multi-Agent Design
**Date:** 2025-10-21 | **Arxiv:** [2510.16652](https://arxiv.org/abs/2510.16652)

#### Abstract
Modern scientific and engineering design increasingly involves distributed optimization, where agents such as laboratories, simulations, or industrial partners pursue related goals under differing conditions. These agents often face heterogeneities in objectives, evaluation budgets, and accessible design variables, which complicates coordination and can lead to redundancy, poor resource use, and ineffective information sharing. Bayesian Optimization (BO) is a widely used decision-making framework for expensive black box functions, but its single-agent formulation assumes centralized control and full data sharing. Recent collaborative BO methods relax these assumptions, yet they often require uniform resources, fully shared input spaces, and fixed task alignment, conditions rarely satisfied in practice. To address these challenges, we introduce Adaptive Resource Aware Collaborative Bayesian Optimization (ARCO-BO), a framework that explicitly accounts for heterogeneity in multi-agent optimization. ARCO-BO combines three components: a similarity and optima-aware consensus mechanism for adaptive information sharing, a budget-aware asynchronous sampling strategy for resource coordination, and a partial input space sharing for heterogeneous design spaces. Experiments on synthetic and high-dimensional engineering problems show that ARCO-BO consistently outperforms independent BO and existing collaborative BO via consensus approach, achieving robust and efficient performance in complex multi-agent settings.

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


### LLM-guided Chemical Process Optimization with a Multi-Agent Approach
**Date:** 2025-10-17 | **Arxiv:** [2506.20921](https://arxiv.org/abs/2506.20921)

#### Abstract
Chemical process optimization maximizes production efficiency and economic performance, but optimization algorithms, including gradient-based solvers, numerical methods, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable. We present a multi-agent LLM framework that autonomously infers operating constraints from minimal process descriptions, then collaboratively guides optimization. Our AutoGen-based framework employs OpenAI's o3 model with specialized agents for constraint generation, parameter validation, simulation, and optimization guidance. Through autonomous constraint generation and iterative multi-agent optimization, the framework eliminates the need for predefined operational bounds. Validated on hydrodealkylation across cost, yield, and yield-to-cost ratio metrics, the framework achieved competitive performance with conventional methods while reducing wall-time 31-fold relative to grid search, converging in under 20 minutes. The reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs and applying domain-informed heuristics. Unlike conventional methods requiring predefined constraints, our approach uniquely combines autonomous constraint generation with interpretable parameter exploration. Model comparison reveals reasoning-capable architectures (o3, o1) are essential for successful optimization, while standard models fail to converge. This approach is particularly valuable for emerging processes and retrofit applications where operational constraints are poorly characterized or unavailable.

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


### MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation
**Date:** 2025-10-17 | **Arxiv:** [2510.14184](https://arxiv.org/abs/2510.14184)

#### Abstract
We present MAFA (Multi-Agent Framework for Annotation), a production-deployed system that transforms enterprise-scale annotation workflows through configurable multi-agent collaboration. Addressing the critical challenge of annotation backlogs in financial services, where millions of customer utterances require accurate categorization, MAFA combines specialized agents with structured reasoning and a judge-based consensus mechanism. Our framework uniquely supports dynamic task adaptation, allowing organizations to define custom annotation types (FAQs, intents, entities, or domain-specific categories) through configuration rather than code changes. Deployed at JP Morgan Chase, MAFA has eliminated a 1 million utterance backlog while achieving, on average, 86% agreement with human annotators, annually saving over 5,000 hours of manual annotation work. The system processes utterances with annotation confidence classifications, which are typically 85% high, 10% medium, and 5% low across all datasets we tested. This enables human annotators to focus exclusively on ambiguous and low-coverage cases. We demonstrate MAFA's effectiveness across multiple datasets and languages, showing consistent improvements over traditional and single-agent annotation baselines: 13.8% higher Top-1 accuracy, 15.1% improvement in Top-5 accuracy, and 16.9% better F1 in our internal intent classification dataset and similar gains on public benchmarks. This work bridges the gap between theoretical multi-agent systems and practical enterprise deployment, providing a blueprint for organizations facing similar annotation challenges.

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


### Multi-Agent Regime-Conditioned Diffusion (MARCD) for CVaR-Constrained Portfolio Decisions
**Date:** 2025-10-15 | **Arxiv:** [2510.10807](https://arxiv.org/abs/2510.10807)

#### Abstract
We examine whether regime-conditioned generative scenarios combined with a convex CVaR allocator improve portfolio decisions under regime shifts. We present MARCD, a generative-to-decision framework with: (i) a Gaussian HMM to infer latent regimes; (ii) a diffusion generator that produces regime-conditioned scenarios; (iii) signal extraction via blended, shrunk moments; and (iv) a governed CVaR epigraph quadratic program. Contributions: Within the Scenario stage we introduce a tail-weighted diffusion objective that up-weights low-quantile outcomes relevant for drawdowns and a regime-expert (MoE) denoiser whose gate increases with crisis posteriors; both are evaluated end-to-end through the allocator. Under strict walk-forward on liquid multi-asset ETFs (2005-2025), MARCD exhibits stronger scenario calibration and materially smaller drawdowns: MaxDD 9.3% versus 14.1% for BL (a 34% reduction) over 2020-2025 out-of-sample. The framework provides an auditable pipeline with explicit budget, box, and turnover constraints, demonstrating the value of decision-aware generative modeling in finance.

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


### $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks
**Date:** 2025-10-10 | **Arxiv:** [2504.00218](https://arxiv.org/abs/2504.00218)

#### Abstract
Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.

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
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Distributed Algorithms for Multi-Agent Multi-Armed Bandits with Collision
**Date:** 2025-10-09 | **Arxiv:** [2510.06683](https://arxiv.org/abs/2510.06683)

#### Abstract
We study the stochastic Multiplayer Multi-Armed Bandit (MMAB) problem, where multiple players select arms to maximize their cumulative rewards. Collisions occur when two or more players select the same arm, resulting in no reward, and are observed by the players involved. We consider a distributed setting without central coordination, where each player can only observe their own actions and collision feedback. We propose a distributed algorithm with an adaptive, efficient communication protocol. The algorithm achieves near-optimal group and individual regret, with a communication cost of only $\mathcal{O}(\log\log T)$. Our experiments demonstrate significant performance improvements over existing baselines. Compared to state-of-the-art (SOTA) methods, our approach achieves a notable reduction in individual regret. Finally, we extend our approach to a periodic asynchronous setting, proving the lower bound for this problem and presenting an algorithm that achieves logarithmic regret.

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


### LLMs as Policy-Agnostic Teammates: A Case Study in Human Proxy Design for Heterogeneous Agent Teams
**Date:** 2025-10-08 | **Arxiv:** [2510.06151](https://arxiv.org/abs/2510.06151)

#### Abstract
A critical challenge in modelling Heterogeneous-Agent Teams is training agents to collaborate with teammates whose policies are inaccessible or non-stationary, such as humans. Traditional approaches rely on expensive human-in-the-loop data, which limits scalability. We propose using Large Language Models (LLMs) as policy-agnostic human proxies to generate synthetic data that mimics human decision-making. To evaluate this, we conduct three experiments in a grid-world capture game inspired by Stag Hunt, a game theory paradigm that balances risk and reward. In Experiment 1, we compare decisions from 30 human participants and 2 expert judges with outputs from LLaMA 3.1 and Mixtral 8x22B models. LLMs, prompted with game-state observations and reward structures, align more closely with experts than participants, demonstrating consistency in applying underlying decision criteria. Experiment 2 modifies prompts to induce risk-sensitive strategies (e.g. "be risk averse"). LLM outputs mirror human participants' variability, shifting between risk-averse and risk-seeking behaviours. Finally, Experiment 3 tests LLMs in a dynamic grid-world where the LLM agents generate movement actions. LLMs produce trajectories resembling human participants' paths. While LLMs cannot yet fully replicate human adaptability, their prompt-guided diversity offers a scalable foundation for simulating policy-agnostic teammates.

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


### Multi-Agent Path Finding via Offline RL and LLM Collaboration
**Date:** 2025-09-29 | **Arxiv:** [2509.22130](https://arxiv.org/abs/2509.22130)

#### Abstract
Multi-Agent Path Finding (MAPF) poses a significant and challenging problem critical for applications in robotics and logistics, particularly due to its combinatorial complexity and the partial observability inherent in realistic environments. Decentralized reinforcement learning methods commonly encounter two substantial difficulties: first, they often yield self-centered behaviors among agents, resulting in frequent collisions, and second, their reliance on complex communication modules leads to prolonged training times, sometimes spanning weeks. To address these challenges, we propose an efficient decentralized planning framework based on the Decision Transformer (DT), uniquely leveraging offline reinforcement learning to substantially reduce training durations from weeks to mere hours. Crucially, our approach effectively handles long-horizon credit assignment and significantly improves performance in scenarios with sparse and delayed rewards. Furthermore, to overcome adaptability limitations inherent in standard RL methods under dynamic environmental changes, we integrate a large language model (GPT-4o) to dynamically guide agent policies. Extensive experiments in both static and dynamically changing environments demonstrate that our DT-based approach, augmented briefly by GPT-4o, significantly enhances adaptability and performance.

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


### A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks
**Date:** 2025-09-19 | **Arxiv:** [2509.14285](https://arxiv.org/abs/2509.14285)

#### Abstract
Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.

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


### Multi-Agent Systems Execute Arbitrary Malicious Code
**Date:** 2025-09-16 | **Arxiv:** [2503.12188](https://arxiv.org/abs/2503.12188)

#### Abstract
Multi-agent systems coordinate LLM-based agents to perform tasks on users' behalf. In real-world applications, multi-agent systems will inevitably interact with untrusted inputs, such as malicious Web content, files, email attachments, and more.   Using several recently proposed multi-agent frameworks as concrete examples, we demonstrate that adversarial content can hijack control and communication within the system to invoke unsafe agents and functionalities. This results in a complete security breach, up to execution of arbitrary malicious code on the user's device or exfiltration of sensitive data from the user's containerized environment. For example, when agents are instantiated with GPT-4o, Web-based attacks successfully cause the multi-agent system execute arbitrary malicious code in 58-90\% of trials (depending on the orchestrator). In some model-orchestrator configurations, the attack success rate is 100\%. We also demonstrate that these attacks succeed even if individual agents are not susceptible to direct or indirect prompt injection, and even if they refuse to perform harmful actions. We hope that these results will motivate development of trust and security models for multi-agent systems before they are widely deployed.

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
* **Layer:** Application
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Self-Supervised Goal-Reaching Results in Multi-Agent Cooperation and Exploration
**Date:** 2025-09-16 | **Arxiv:** [2509.10656](https://arxiv.org/abs/2509.10656)

#### Abstract
For groups of autonomous agents to achieve a particular goal, they must engage in coordination and long-horizon reasoning. However, designing reward functions to elicit such behavior is challenging. In this paper, we study how self-supervised goal-reaching techniques can be leveraged to enable agents to cooperate. The key idea is that, rather than have agents maximize some scalar reward, agents aim to maximize the likelihood of visiting a certain goal. This problem setting enables human users to specify tasks via a single goal state rather than implementing a complex reward function. While the feedback signal is quite sparse, we will demonstrate that self-supervised goal-reaching techniques enable agents to learn from such feedback. On MARL benchmarks, our proposed method outperforms alternative approaches that have access to the same sparse reward signal as our method. While our method has no explicit mechanism for exploration, we observe that self-supervised multi-agent goal-reaching leads to emergent cooperation and exploration in settings where alternative approaches never witness a single successful trial.

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
* **Limits:** However, designing reward functions to elicit such behavior is challenging.
* **Signal Tags:** #ai

---


### Scene-Aware Vectorized Memory Multi-Agent Framework with Cross-Modal Differentiated Quantization VLMs for Visually Impaired Assistance
**Date:** 2025-08-26 | **Arxiv:** [2508.18177](https://arxiv.org/abs/2508.18177)

#### Abstract
This study proposes the dual technological innovation framework, including a cross-modal differ entiated quantization framework for vision-language models (VLMs) and a scene-aware vectorized   memory multi-agent system for visually impaired assistance. The modular framework was developed   implementing differentiated processing strategies, effectively reducing memory requirements from   38GB to 16GB while maintaining model performance. The multi-agent architecture combines   scene classification, vectorized memory, and multimodal interaction, enabling persistent storage   and efficient retrieval of scene memories. Through perception-memory-reasoning workflows, the   system provides environmental information beyond the current view using historical memories.   Experiments show the quantized 19B-parameter model only experiences a 2.05% performance drop   on MMBench and maintains 63.7 accuracy on OCR-VQA (original: 64.9), outperforming smaller   models with equivalent memory requirements like the Molmo-7B series. The system maintains   response latency between 2.83-3.52 seconds from scene analysis to initial speech output, substantially   faster than non-streaming methods. This research advances computational efficiency and assistive   technology, offering visually impaired users comprehensive real-time assistance in scene perception,   text recognition, and navigation.

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


### JARVIS: A Multi-Agent Code Assistant for High-Quality EDA Script Generation
**Date:** 2025-08-19 | **Arxiv:** [2505.14978](https://arxiv.org/abs/2505.14978)

#### Abstract
This paper presents JARVIS, a novel multi-agent framework that leverages Large Language Models (LLMs) and domain expertise to generate high-quality scripts for specialized Electronic Design Automation (EDA) tasks. By combining a domain-specific LLM trained with synthetically generated data, a custom compiler for structural verification, rule enforcement, code fixing capabilities, and advanced retrieval mechanisms, our approach achieves significant improvements over state-of-the-art domain-specific models. Our framework addresses the challenges of data scarcity and hallucination errors in LLMs, demonstrating the potential of LLMs in specialized engineering domains. We evaluate our framework on multiple benchmarks and show that it outperforms existing models in terms of accuracy and reliability. Our work sets a new precedent for the application of LLMs in EDA and paves the way for future innovations in this field.

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


### A Unified Multi-Agent Framework for Universal Multimodal Understanding and Generation
**Date:** 2025-08-15 | **Arxiv:** [2508.10494](https://arxiv.org/abs/2508.10494)

#### Abstract
Real-world multimodal applications often require any-to-any capabilities, enabling both understanding and generation across modalities including text, image, audio, and video. However, integrating the strengths of autoregressive language models (LLMs) for reasoning and diffusion models for high-fidelity generation remains challenging. Existing approaches rely on rigid pipelines or tightly coupled architectures, limiting flexibility and scalability. We propose MAGUS (Multi-Agent Guided Unified Multimodal System), a modular framework that unifies multimodal understanding and generation via two decoupled phases: Cognition and Deliberation. MAGUS enables symbolic multi-agent collaboration within a shared textual workspace. In the Cognition phase, three role-conditioned multimodal LLM agents - Perceiver, Planner, and Reflector - engage in collaborative dialogue to perform structured understanding and planning. The Deliberation phase incorporates a Growth-Aware Search mechanism that orchestrates LLM-based reasoning and diffusion-based generation in a mutually reinforcing manner. MAGUS supports plug-and-play extensibility, scalable any-to-any modality conversion, and semantic alignment - all without the need for joint training. Experiments across multiple benchmarks, including image, video, and audio generation, as well as cross-modal instruction following, demonstrate that MAGUS outperforms strong baselines and state-of-the-art systems. Notably, on the MME benchmark, MAGUS surpasses the powerful closed-source model GPT-4o.

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
* **Layer:** Application
* **Limits:** However, integrating the strengths of autoregressive language models (LLMs) for reasoning and diffusion models for high-fidelity generation remains challenging.
* **Signal Tags:** #ai

---
