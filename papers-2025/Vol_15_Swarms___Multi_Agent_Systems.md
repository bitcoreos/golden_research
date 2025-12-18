# Vol 15 Swarms   Multi Agent Systems
*Enriched by BITCOREOS | Phase 4 Batch 3*

---

### Multi-Agent collaboration patterns with Strands Agents and Amazon Nova
**Date:** 2025-11-11 | **Arxiv:** [](https://arxiv.org/abs/)

#### Abstract
Multi-agent generative AI systems use multiple specialized AI agents working together to handle complex, multi-faceted tasks that exceed the capabilities of any single model. By combining agents with different skills or modalities (for example, langu...

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


### Transform your MCP architecture: Unite MCP servers through AgentCore Gateway
**Date:** 2025-11-06 | **Arxiv:** [](https://arxiv.org/abs/)

#### Abstract
As AI agents are adopted at scale, developer teams can create dozens to hundreds of specialized Model Context Protocol (MCP) servers, tailored for specific agent use case and domain, organization functions or teams. Organizations also need to integra...

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


### DR. WELL: Dynamic Reasoning and Learning with Symbolic World Model for Embodied LLM-Based Multi-Agent Collaboration
**Date:** 2025-11-07 | **Arxiv:** [2511.04646](https://arxiv.org/abs/2511.04646)

#### Abstract
Cooperative multi-agent planning requires agents to make joint decisions with partial information and limited communication. Coordination at the trajectory level often fails, as small deviations in timing or movement cascade into conflicts. Symbolic planning mitigates this challenge by raising the level of abstraction and providing a minimal vocabulary of actions that enable synchronization and collective progress. We present DR. WELL, a decentralized neurosymbolic framework for cooperative multi-agent planning. Cooperation unfolds through a two-phase negotiation protocol: agents first propose candidate roles with reasoning and then commit to a joint allocation under consensus and environment constraints. After commitment, each agent independently generates and executes a symbolic plan for its role without revealing detailed trajectories. Plans are grounded in execution outcomes via a shared world model that encodes the current state and is updated as agents act. By reasoning over symbolic plans rather than raw trajectories, DR. WELL avoids brittle step-level alignment and enables higher-level operations that are reusable, synchronizable, and interpretable. Experiments on cooperative block-push tasks show that agents adapt across episodes, with the dynamic world model capturing reusable patterns and improving task completion rates and efficiency. Experiments on cooperative block-push tasks show that our dynamic world model improves task completion and efficiency through negotiation and self-refinement, trading a time overhead for evolving, more efficient collaboration strategies.

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


### Build multi-agent site reliability engineering assistants with Amazon Bedrock AgentCore
**Date:** 2025-09-26 | **Arxiv:** [](https://arxiv.org/abs/)

#### Abstract
Site reliability engineers (SREs) face an increasingly complex challenge in modern distributed systems. During production incidents, they must rapidly correlate data from multiple sources—logs, metrics, Kubernetes events, and operational runbooks—to ...

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


### Multi-Agent Pointer Transformer: Seq-to-Seq Reinforcement Learning for Multi-Vehicle Dynamic Pickup-Delivery Problems
**Date:** 2025-11-24 | **Arxiv:** [2511.17435](https://arxiv.org/abs/2511.17435)

#### Abstract
This paper addresses the cooperative Multi-Vehicle Dynamic Pickup and Delivery Problem with Stochastic Requests (MVDPDPSR) and proposes an end-to-end centralized decision-making framework based on sequence-to-sequence, named Multi-Agent Pointer Transformer (MAPT). MVDPDPSR is an extension of the vehicle routing problem and a spatio-temporal system optimization problem, widely applied in scenarios such as on-demand delivery. Classical operations research methods face bottlenecks in computational complexity and time efficiency when handling large-scale dynamic problems. Although existing reinforcement learning methods have achieved some progress, they still encounter several challenges: 1) Independent decoding across multiple vehicles fails to model joint action distributions; 2) The feature extraction network struggles to capture inter-entity relationships; 3) The joint action space is exponentially large. To address these issues, we designed the MAPT framework, which employs a Transformer Encoder to extract entity representations, combines a Transformer Decoder with a Pointer Network to generate joint action sequences in an AutoRegressive manner, and introduces a Relation-Aware Attention module to capture inter-entity relationships. Additionally, we guide the model's decision-making using informative priors to facilitate effective exploration. Experiments on 8 datasets demonstrate that MAPT significantly outperforms existing baseline methods in terms of performance and exhibits substantial computational time advantages compared to classical operations research methods.

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


### Transformer-Based Scalable Multi-Agent Reinforcement Learning for Networked Systems with Long-Range Interactions
**Date:** 2025-11-18 | **Arxiv:** [2511.13103](https://arxiv.org/abs/2511.13103)

#### Abstract
Multi-agent reinforcement learning (MARL) has shown promise for large-scale network control, yet existing methods face two major limitations. First, they typically rely on assumptions leading to decay properties of local agent interactions, limiting their ability to capture long-range dependencies such as cascading power failures or epidemic outbreaks. Second, most approaches lack generalizability across network topologies, requiring retraining when applied to new graphs. We introduce STACCA (Shared Transformer Actor-Critic with Counterfactual Advantage), a unified transformer-based MARL framework that addresses both challenges. STACCA employs a centralized Graph Transformer Critic to model long-range dependencies and provide system-level feedback, while its shared Graph Transformer Actor learns a generalizable policy capable of adapting across diverse network structures. Further, to improve credit assignment during training, STACCA integrates a novel counterfactual advantage estimator that is compatible with state-value critic estimates. We evaluate STACCA on epidemic containment and rumor-spreading network control tasks, demonstrating improved performance, network generalization, and scalability. These results highlight the potential of transformer-based MARL architectures to achieve scalable and generalizable control in large-scale networked systems.

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


### Large-scale automatic carbon ion treatment planning for head and neck cancers via parallel multi-agent reinforcement learning
**Date:** 2025-11-05 | **Arxiv:** [2511.02314](https://arxiv.org/abs/2511.02314)

#### Abstract
Head-and-neck cancer (HNC) planning is difficult because multiple critical organs-at-risk (OARs) are close to complex targets. Intensity-modulated carbon-ion therapy (IMCT) offers superior dose conformity and OAR sparing but remains slow due to relative biological effectiveness (RBE) modeling, leading to laborious, experience-based, and often suboptimal tuning of many treatment-planning parameters (TPPs). Recent deep learning (DL) methods are limited by data bias and plan feasibility, while reinforcement learning (RL) struggles to efficiently explore the exponentially large TPP search space. We propose a scalable multi-agent RL (MARL) framework for parallel tuning of 45 TPPs in IMCT. It uses a centralized-training decentralized-execution (CTDE) QMIX backbone with Double DQN, Dueling DQN, and recurrent encoding (DRQN) for stable learning in a high-dimensional, non-stationary environment. To enhance efficiency, we (1) use compact historical DVH vectors as state inputs, (2) apply a linear action-to-value transform mapping small discrete actions to uniform parameter adjustments, and (3) design an absolute, clinically informed piecewise reward aligned with plan scores. A synchronous multi-process worker system interfaces with the PHOENIX TPS for parallel optimization and accelerated data collection. On a head-and-neck dataset (10 training, 10 testing), the method tuned 45 parameters simultaneously and produced plans comparable to or better than expert manual ones (relative plan score: RL $85.93\pm7.85%$ vs Manual $85.02\pm6.92%$), with significant (p-value $<$ 0.05) improvements for five OARs. The framework efficiently explores high-dimensional TPP spaces and generates clinically competitive IMCT plans through direct TPS interaction, notably improving OAR sparing.

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


### MAGIC-MASK: Multi-Agent Guided Inter-Agent Collaboration with Mask-Based Explainability for Reinforcement Learning
**Date:** 2025-10-02 | **Arxiv:** [2510.00274](https://arxiv.org/abs/2510.00274)

#### Abstract
Understanding the decision-making process of Deep Reinforcement Learning agents remains a key challenge for deploying these systems in safety-critical and multi-agent environments. While prior explainability methods like StateMask, have advanced the identification of critical states, they remain limited by computational cost, exploration coverage, and lack of adaptation to multi-agent settings. To overcome these limitations, we propose a mathematically grounded framework, MAGIC-MASK (Multi-Agent Guided Inter-agent Collaboration with Mask-Based Explainability for Reinforcement Learning), that extends perturbation-based explanation to Multi-Agent Reinforcement Learning. Our method integrates Proximal Policy Optimization, adaptive epsilon-greedy exploration, and lightweight inter-agent collaboration to share masked state information and peer experience. This collaboration enables each agent to perform saliency-guided masking and share reward-based insights with peers, reducing the time required for critical state discovery, improving explanation fidelity, and leading to faster and more robust learning. The core novelty of our approach lies in generalizing explainability from single-agent to multi-agent systems through a unified mathematical formalism built on trajectory perturbation, reward fidelity analysis, and Kullback-Leibler divergence regularization. This framework yields localized, interpretable explanations grounded in probabilistic modeling and multi-agent Markov decision processes. We validate our framework on both single-agent and multi-agent benchmarks, including a multi-agent highway driving environment and Google Research Football, demonstrating that MAGIC-MASK consistently outperforms state-of-the-art baselines in fidelity, learning efficiency, and policy robustness while offering interpretable and transferable explanations.

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


### Toward Autonomous Engineering Design: A Knowledge-Guided Multi-Agent Framework
**Date:** 2025-11-06 | **Arxiv:** [2511.03179](https://arxiv.org/abs/2511.03179)

#### Abstract
The engineering design process often demands expertise from multiple domains, leading to complex collaborations and iterative refinements. Traditional methods can be resource-intensive and prone to inefficiencies. To address this, we formalize the engineering design process through a multi-agent AI framework that integrates structured design and review loops. The framework introduces specialized knowledge-driven agents that collaborate to generate and refine design candidates. As an exemplar, we demonstrate its application to the aerodynamic optimization of 4-digit NACA airfoils. The framework consists of three key AI agents: a Graph Ontologist, a Design Engineer, and a Systems Engineer. The Graph Ontologist employs a Large Language Model (LLM) to construct two domain-specific knowledge graphs from airfoil design literature. The Systems Engineer, informed by a human manager, formulates technical requirements that guide design generation and evaluation. The Design Engineer leverages the design knowledge graph and computational tools to propose candidate airfoils meeting these requirements. The Systems Engineer reviews and provides feedback both qualitative and quantitative using its own knowledge graph, forming an iterative feedback loop until a design is validated by the manager. The final design is then optimized to maximize performance metrics such as the lift-to-drag ratio. Overall, this work demonstrates how collaborative AI agents equipped with structured knowledge representations can enhance efficiency, consistency, and quality in the engineering design process.

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


### From Solo to Symphony: Orchestrating Multi-Agent Collaboration with Single-Agent Demos
**Date:** 2025-11-05 | **Arxiv:** [2511.02762](https://arxiv.org/abs/2511.02762)

#### Abstract
Training a team of agents from scratch in multi-agent reinforcement learning (MARL) is highly inefficient, much like asking beginners to play a symphony together without first practicing solo. Existing methods, such as offline or transferable MARL, can ease this burden, but they still rely on costly multi-agent data, which often becomes the bottleneck. In contrast, solo experiences are far easier to obtain in many important scenarios, e.g., collaborative coding, household cooperation, and search-and-rescue. To unlock their potential, we propose Solo-to-Collaborative RL (SoCo), a framework that transfers solo knowledge into cooperative learning. SoCo first pretrains a shared solo policy from solo demonstrations, then adapts it for cooperation during multi-agent training through a policy fusion mechanism that combines an MoE-like gating selector and an action editor. Experiments across diverse cooperative tasks show that SoCo significantly boosts the training efficiency and performance of backbone algorithms. These results demonstrate that solo demonstrations provide a scalable and effective complement to multi-agent data, making cooperative learning more practical and broadly applicable.

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


### Learning Decentralized Routing Policies via Graph Attention-based Multi-Agent Reinforcement Learning in Lunar Delay-Tolerant Networks
**Date:** 2025-10-24 | **Arxiv:** [2510.20436](https://arxiv.org/abs/2510.20436)

#### Abstract
We present a fully decentralized routing framework for multi-robot exploration missions operating under the constraints of a Lunar Delay-Tolerant Network (LDTN). In this setting, autonomous rovers must relay collected data to a lander under intermittent connectivity and unknown mobility patterns. We formulate the problem as a Partially Observable Markov Decision Problem (POMDP) and propose a Graph Attention-based Multi-Agent Reinforcement Learning (GAT-MARL) policy that performs Centralized Training, Decentralized Execution (CTDE). Our method relies only on local observations and does not require global topology updates or packet replication, unlike classical approaches such as shortest path and controlled flooding-based algorithms. Through Monte Carlo simulations in randomized exploration environments, GAT-MARL provides higher delivery rates, no duplications, and fewer packet losses, and is able to leverage short-term mobility forecasts; offering a scalable solution for future space robotic systems for planetary exploration, as demonstrated by successful generalization to larger rover teams.

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


### Verification-Aware Planning for Multi-Agent Systems
**Date:** 2025-10-21 | **Arxiv:** [2510.17109](https://arxiv.org/abs/2510.17109)

#### Abstract
Large language model (LLM) agents are increasingly deployed to tackle complex tasks, often necessitating collaboration among multiple specialized agents. However, multi-agent collaboration introduces new challenges in planning, coordination, and verification. Execution failures frequently arise not from flawed reasoning alone, but from subtle misalignments in task interpretation, output format, or inter-agent handoffs. To address these challenges, we present VeriMAP, a framework for multi-agent collaboration with verification-aware planning. The VeriMAP planner decomposes tasks, models subtask dependencies, and encodes planner-defined passing criteria as subtask verification functions (VFs) in Python and natural language. We evaluate VeriMAP on diverse datasets, demonstrating that it outperforms both single- and multi-agent baselines while enhancing system robustness and interpretability. Our analysis highlights how verification-aware planning enables reliable coordination and iterative refinement in multi-agent systems, without relying on external labels or annotations.

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
* **Limits:** However, multi-agent collaboration introduces new challenges in planning, coordination, and verification.
* **Signal Tags:** #ai

---


### Benefits and Limitations of Communication in Multi-Agent Reasoning
**Date:** 2025-10-17 | **Arxiv:** [2510.13903](https://arxiv.org/abs/2510.13903)

#### Abstract
Chain-of-thought prompting has popularized step-by-step reasoning in large language models, yet model performance still degrades as problem complexity and context length grow. By decomposing difficult tasks with long contexts into shorter, manageable ones, recent multi-agent paradigms offer a promising near-term solution to this problem. However, the fundamental capacities of such systems are poorly understood. In this work, we propose a theoretical framework to analyze the expressivity of multi-agent systems. We apply our framework to three algorithmic families: state tracking, recall, and $k$-hop reasoning. We derive bounds on (i) the number of agents required to solve the task exactly, (ii) the quantity and structure of inter-agent communication, and (iii) the achievable speedups as problem size and context scale. Our results identify regimes where communication is provably beneficial, delineate tradeoffs between agent count and bandwidth, and expose intrinsic limitations when either resource is constrained. We complement our theoretical analysis with a set of experiments on pretrained LLMs using controlled synthetic benchmarks. Empirical outcomes confirm the tradeoffs between key quantities predicted by our theory. Collectively, our analysis offers principled guidance for designing scalable multi-agent reasoning systems.

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
* **Limits:** However, the fundamental capacities of such systems are poorly understood.
* **Signal Tags:** #ai

---


### Learning Equilibria from Data: Provably Efficient Multi-Agent Imitation Learning
**Date:** 2025-10-10 | **Arxiv:** [2505.17610](https://arxiv.org/abs/2505.17610)

#### Abstract
This paper provides the first expert sample complexity characterization for learning a Nash equilibrium from expert data in Markov Games. We show that a new quantity named the single policy deviation concentrability coefficient is unavoidable in the non-interactive imitation learning setting, and we provide an upper bound for behavioral cloning (BC) featuring such coefficient. BC exhibits substantial regret in games with high concentrability coefficient, leading us to utilize expert queries to develop and introduce two novel solution algorithms: MAIL-BRO and MURMAIL. The former employs a best response oracle and learns an $\varepsilon$-Nash equilibrium with $\mathcal{O}(\varepsilon^{-4})$ expert and oracle queries. The latter bypasses completely the best response oracle at the cost of a worse expert query complexity of order $\mathcal{O}(\varepsilon^{-8})$. Finally, we provide numerical evidence, confirming our theoretical findings.

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


### ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems
**Date:** 2025-10-08 | **Arxiv:** [2510.05746](https://arxiv.org/abs/2510.05746)

#### Abstract
Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization.

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
* **Limits:** However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines.
* **Signal Tags:** #ai

---


### Language-Driven Coordination and Learning in Multi-Agent Simulation Environments
**Date:** 2025-08-20 | **Arxiv:** [2506.04251](https://arxiv.org/abs/2506.04251)

#### Abstract
This paper introduces LLM-MARL, a unified framework that incorporates large language models (LLMs) into multi-agent reinforcement learning (MARL) to enhance coordination, communication, and generalization in simulated game environments. The framework features three modular components of Coordinator, Communicator, and Memory, which dynamically generate subgoals, facilitate symbolic inter-agent messaging, and support episodic recall. Training combines PPO with a language-conditioned loss and LLM query gating. LLM-MARL is evaluated in Google Research Football, MAgent Battle, and StarCraft II. Results show consistent improvements over MAPPO and QMIX in win rate, coordination score, and zero-shot generalization. Ablation studies demonstrate that subgoal generation and language-based messaging each contribute significantly to performance gains. Qualitative analysis reveals emergent behaviors such as role specialization and communication-driven tactics. By bridging language modeling and policy learning, this work contributes to the design of intelligent, cooperative agents in interactive simulations. It offers a path forward for leveraging LLMs in multi-agent systems used for training, games, and human-AI collaboration.

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


### Multi-agent learning under uncertainty: Recurrence vs. concentration
**Date:** 2025-12-10 | **Arxiv:** [2512.08132](https://arxiv.org/abs/2512.08132)

#### Abstract
In this paper, we examine the convergence landscape of multi-agent learning under uncertainty. Specifically, we analyze two stochastic models of regularized learning in continuous games -- one in continuous and one in discrete time with the aim of characterizing the long-run behavior of the induced sequence of play. In stark contrast to deterministic, full-information models of learning (or models with a vanishing learning rate), we show that the resulting dynamics do not converge in general. In lieu of this, we ask instead which actions are played more often in the long run, and by how much. We show that, in strongly monotone games, the dynamics of regularized learning may wander away from equilibrium infinitely often, but they always return to its vicinity in finite time (which we estimate), and their long-run distribution is sharply concentrated around a neighborhood thereof. We quantify the degree of this concentration, and we show that these favorable properties may all break down if the underlying game is not strongly monotone -- underscoring in this way the limits of regularized learning in the presence of persistent randomness and uncertainty.

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


### Multi-Agent Deep Reinforcement Learning for Collaborative UAV Relay Networks under Jamming Atatcks
**Date:** 2025-12-10 | **Arxiv:** [2512.08341](https://arxiv.org/abs/2512.08341)

#### Abstract
The deployment of Unmanned Aerial Vehicle (UAV) swarms as dynamic communication relays is critical for next-generation tactical networks. However, operating in contested environments requires solving a complex trade-off, including maximizing system throughput while ensuring collision avoidance and resilience against adversarial jamming. Existing heuristic-based approaches often struggle to find effective solutions due to the dynamic and multi-objective nature of this problem. This paper formulates this challenge as a cooperative Multi-Agent Reinforcement Learning (MARL) problem, solved using the Centralized Training with Decentralized Execution (CTDE) framework. Our approach employs a centralized critic that uses global state information to guide decentralized actors which operate using only local observations. Simulation results show that our proposed framework significantly outperforms heuristic baselines, increasing the total system throughput by approximately 50% while simultaneously achieving a near-zero collision rate. A key finding is that the agents develop an emergent anti-jamming strategy without explicit programming. They learn to intelligently position themselves to balance the trade-off between mitigating interference from jammers and maintaining effective communication links with ground users.

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
* **Limits:** However, operating in contested environments requires solving a complex trade-off, including maximizing system throughput while ensuring collision avoidance and resilience against adversarial jamming.
* **Signal Tags:** #ai

---


### Multi-Agent Reinforcement Learning for Intraday Operating Rooms Scheduling under Uncertainty
**Date:** 2025-12-05 | **Arxiv:** [2512.04918](https://arxiv.org/abs/2512.04918)

#### Abstract
Intraday surgical scheduling is a multi-objective decision problem under uncertainty-balancing elective throughput, urgent and emergency demand, delays, sequence-dependent setups, and overtime. We formulate the problem as a cooperative Markov game and propose a multi-agent reinforcement learning (MARL) framework in which each operating room (OR) is an agent trained with centralized training and decentralized execution. All agents share a policy trained via Proximal Policy Optimization (PPO), which maps rich system states to actions, while a within-epoch sequential assignment protocol constructs conflict-free joint schedules across ORs. A mixed-integer pre-schedule provides reference starting times for electives; we impose type-specific quadratic delay penalties relative to these references and a terminal overtime penalty, yielding a single reward that captures throughput, timeliness, and staff workload. In simulations reflecting a realistic hospital mix (six ORs, eight surgery types, random urgent and emergency arrivals), the learned policy outperforms six rule-based heuristics across seven metrics and three evaluation subsets, and, relative to an ex post MIP oracle, quantifies optimality gaps. Policy analytics reveal interpretable behavior-prioritizing emergencies, batching similar cases to reduce setups, and deferring lower-value electives. We also derive a suboptimality bound for the sequential decomposition under simplifying assumptions. We discuss limitations-including OR homogeneity and the omission of explicit staffing constraints-and outline extensions. Overall, the approach offers a practical, interpretable, and tunable data-driven complement to optimization for real-time OR scheduling.

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


### Emergent Coordination and Phase Structure in Independent Multi-Agent Reinforcement Learning
**Date:** 2025-12-01 | **Arxiv:** [2511.23315](https://arxiv.org/abs/2511.23315)

#### Abstract
A clearer understanding of when coordination emerges, fluctuates, or collapses in decentralized multi-agent reinforcement learning (MARL) is increasingly sought in order to characterize the dynamics of multi-agent learning systems. We revisit fully independent Q-learning (IQL) as a minimal decentralized testbed and run large-scale experiments across environment size L and agent density rho. We construct a phase map using two axes - the cooperative success rate (CSR) and a stability index derived from TD-error variance - revealing three distinct regimes: a coordinated and stable phase, a fragile transition region, and a jammed or disordered phase. A sharp double Instability Ridge separates these regimes and corresponds to persistent kernel drift, the time-varying shift of each agent's effective transition kernel induced by others' policy updates. Synchronization analysis further shows that temporal alignment is required for sustained cooperation, and that competition between drift and synchronization generates the fragile regime. Removing agent identifiers eliminates drift entirely and collapses the three-phase structure, demonstrating that small inter-agent asymmetries are a necessary driver of drift. Overall, the results show that decentralized MARL exhibits a coherent phase structure governed by the interaction between scale, density, and kernel drift, suggesting that emergent coordination behaves as a distribution-interaction-driven phase phenomenon.

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


### MOMA-AC: A preference-driven actor-critic framework for continuous multi-objective multi-agent reinforcement learning
**Date:** 2025-11-25 | **Arxiv:** [2511.18181](https://arxiv.org/abs/2511.18181)

#### Abstract
This paper addresses a critical gap in Multi-Objective Multi-Agent Reinforcement Learning (MOMARL) by introducing the first dedicated inner-loop actor-critic framework for continuous state and action spaces: Multi-Objective Multi-Agent Actor-Critic (MOMA-AC). Building on single-objective, single-agent algorithms, we instantiate this framework with Twin Delayed Deep Deterministic Policy Gradient (TD3) and Deep Deterministic Policy Gradient (DDPG), yielding MOMA-TD3 and MOMA-DDPG. The framework combines a multi-headed actor network, a centralised critic, and an objective preference-conditioning architecture, enabling a single neural network to encode the Pareto front of optimal trade-off policies for all agents across conflicting objectives in a continuous MOMARL setting. We also outline a natural test suite for continuous MOMARL by combining a pre-existing multi-agent single-objective physics simulator with its multi-objective single-agent counterpart. Evaluating cooperative locomotion tasks in this suite, we show that our framework achieves statistically significant improvements in expected utility and hypervolume relative to outer-loop and independent training baselines, while demonstrating stable scalability as the number of agents increases. These results establish our framework as a foundational step towards robust, scalable multi-objective policy learning in continuous multi-agent domains.

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


### Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration
**Date:** 2025-11-25 | **Arxiv:** [2511.19417](https://arxiv.org/abs/2511.19417)

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in challenging, knowledge-intensive reasoning tasks. However, extending LLMs to perceive and reason over a new modality (e.g., vision), often requires costly development of large-scale vision language models (VLMs) with LLMs as backbones. Smaller VLMs are more efficient and adaptable but often lack the broad knowledge and reasoning capabilities of frontier LLMs. In this work, we propose BeMyEyes, a modular, multi-agent framework for extending LLMs to multimodal reasoning by orchestrating collaboration between efficient, adaptable VLMs as perceivers and powerful LLMs as reasoners through conversations. We then introduce a data synthesis and supervised fine-tuning pipeline to train the perceiver agent to effectively collaborate with the reasoner agent. By combining the complementary strengths of perception and reasoning agents, BeMyEyes avoids the need for training large-scale multimodal models, preserves the generalization and reasoning capabilities of LLMs, and allows flexible extension to new domains and modalities. Experiments show that our framework unlocks the multimodal reasoning capabilities for LLMs, enabling a lightweight and fully open-source solution, i.e. equipping text-only DeepSeek-R1 with Qwen2.5-VL-7B perceiver, to outperform large-scale proprietary VLMs such as GPT-4o on a wide range of knowledge-intensive multimodal tasks. These results demonstrate the effectiveness, modularity, and scalability of our multi-agent approach for building future multimodal reasoning systems.

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
* **Limits:** However, extending LLMs to perceive and reason over a new modality (e.
* **Signal Tags:** #ai

---


### ToC: Tree-of-Claims Search with Multi-Agent Language Models
**Date:** 2025-11-24 | **Arxiv:** [2511.16972](https://arxiv.org/abs/2511.16972)

#### Abstract
Optimizing patent claims is a critical yet challenging task, demanding careful balance between maximizing novelty and preserving legal scope. Manual claim drafting is labor-intensive, costly, and inherently inconsistent, while conventional Large Language Models (LLMs) often lack the structured, iterative reasoning essential for precise claim refinement. To address these challenges, we introduce Tree of Claims (ToC), an innovative framework that redefines claim editing as a guided search problem. ToC synergistically integrates Monte Carlo Tree Search (MCTS) with a collaborative multi-agent system, comprising an LLM-based EditorAgent that proposes contextually grounded edits, and an ExaminerAgent that mimics patent examiner critiques through structured, chain-of-thought analyses of novelty and prior art disclosure. Driven by a carefully designed multi-objective reward function, ToC jointly optimizes novelty, scope retention, and semantic coherence. Experimental evaluation on a benchmark of 1145 claims demonstrates that ToC significantly outperforms standard LLMs in zero-shot and few-shot scenarios, achieving an average composite score improvement of 8\%, and up to 9\% in certain cases. Extensive experiments, including detailed ablation studies, validate ToC's efficacy in generating superior, legally robust claim revisions. Overall, ToC establishes a transparent, controllable, and interpretable methodology that effectively bridges advanced LLM reasoning capabilities with strategic MCTS planning for structured patent claim optimization.The source code is available at https://github.com/ysy2003/ToC.

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


### Beyond Monotonicity: Revisiting Factorization Principles in Multi-Agent Q-Learning
**Date:** 2025-11-14 | **Arxiv:** [2511.09792](https://arxiv.org/abs/2511.09792)

#### Abstract
Value decomposition is a central approach in multi-agent reinforcement learning (MARL), enabling centralized training with decentralized execution by factorizing the global value function into local values. To ensure individual-global-max (IGM) consistency, existing methods either enforce monotonicity constraints, which limit expressive power, or adopt softer surrogates at the cost of algorithmic complexity. In this work, we present a dynamical systems analysis of non-monotonic value decomposition, modeling learning dynamics as continuous-time gradient flow. We prove that, under approximately greedy exploration, all zero-loss equilibria violating IGM consistency are unstable saddle points, while only IGM-consistent solutions are stable attractors of the learning dynamics. Extensive experiments on both synthetic matrix games and challenging MARL benchmarks demonstrate that unconstrained, non-monotonic factorization reliably recovers IGM-optimal solutions and consistently outperforms monotonic baselines. Additionally, we investigate the influence of temporal-difference targets and exploration strategies, providing actionable insights for the design of future value-based MARL algorithms.

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


### TIGER-MARL: Enhancing Multi-Agent Reinforcement Learning with Temporal Information through Graph-based Embeddings and Representations
**Date:** 2025-11-13 | **Arxiv:** [2511.08832](https://arxiv.org/abs/2511.08832)

#### Abstract
In this paper, we propose capturing and utilizing \textit{Temporal Information through Graph-based Embeddings and Representations} or \textbf{TIGER} to enhance multi-agent reinforcement learning (MARL). We explicitly model how inter-agent coordination structures evolve over time. While most MARL approaches rely on static or per-step relational graphs, they overlook the temporal evolution of interactions that naturally arise as agents adapt, move, or reorganize cooperation strategies. Capturing such evolving dependencies is key to achieving robust and adaptive coordination. To this end, TIGER constructs dynamic temporal graphs of MARL agents, connecting their current and historical interactions. It then employs a temporal attention-based encoder to aggregate information across these structural and temporal neighborhoods, yielding time-aware agent embeddings that guide cooperative policy learning. Through extensive experiments on two coordination-intensive benchmarks, we show that TIGER consistently outperforms diverse value-decomposition and graph-based MARL baselines in task performance and sample efficiency. Furthermore, we conduct comprehensive ablation studies to isolate the impact of key design parameters in TIGER, revealing how structural and temporal factors can jointly shape effective policy learning in MARL. All codes can be found here: https://github.com/Nikunj-Gupta/tiger-marl.

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


### Hybrid Training for Enhanced Multi-task Generalization in Multi-agent Reinforcement Learning
**Date:** 2025-11-11 | **Arxiv:** [2408.13567](https://arxiv.org/abs/2408.13567)

#### Abstract
In multi-agent reinforcement learning (MARL), achieving multi-task generalization to diverse agents and objectives presents significant challenges. Existing online MARL algorithms primarily focus on single-task performance, but their lack of multi-task generalization capabilities typically results in substantial computational waste and limited real-life applicability. Meanwhile, existing offline multi-task MARL approaches are heavily dependent on data quality, often resulting in poor performance on unseen tasks. In this paper, we introduce HyGen, a novel hybrid MARL framework, Hybrid Training for Enhanced Multi-Task Generalization, which integrates online and offline learning to ensure both multi-task generalization and training efficiency. Specifically, our framework extracts potential general skills from offline multi-task datasets. We then train policies to select the optimal skills under the centralized training and decentralized execution paradigm (CTDE). During this stage, we utilize a replay buffer that integrates both offline data and online interactions. We empirically demonstrate that our framework effectively extracts and refines general skills, yielding impressive generalization to unseen tasks. Comparative analyses on the StarCraft multi-agent challenge show that HyGen outperforms a wide range of existing solely online and offline methods.

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


### Adaptive Context Length Optimization with Low-Frequency Truncation for Multi-Agent Reinforcement Learning
**Date:** 2025-10-31 | **Arxiv:** [2510.26389](https://arxiv.org/abs/2510.26389)

#### Abstract
Recently, deep multi-agent reinforcement learning (MARL) has demonstrated promising performance for solving challenging tasks, such as long-term dependencies and non-Markovian environments. Its success is partly attributed to conditioning policies on large fixed context length. However, such large fixed context lengths may lead to limited exploration efficiency and redundant information. In this paper, we propose a novel MARL framework to obtain adaptive and effective contextual information. Specifically, we design a central agent that dynamically optimizes context length via temporal gradient analysis, enhancing exploration to facilitate convergence to global optima in MARL. Furthermore, to enhance the adaptive optimization capability of the context length, we present an efficient input representation for the central agent, which effectively filters redundant information. By leveraging a Fourier-based low-frequency truncation method, we extract global temporal trends across decentralized agents, providing an effective and efficient representation of the MARL environment. Extensive experiments demonstrate that the proposed method achieves state-of-the-art (SOTA) performance on long-term dependency tasks, including PettingZoo, MiniGrid, Google Research Football (GRF), and StarCraft Multi-Agent Challenge v2 (SMACv2).

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
* **Limits:** However, such large fixed context lengths may lead to limited exploration efficiency and redundant information.
* **Signal Tags:** #ai

---


### Optimal Information Combining for Multi-Agent Systems Using Adaptive Bias Learning
**Date:** 2025-10-31 | **Arxiv:** [2510.25793](https://arxiv.org/abs/2510.25793)

#### Abstract
Modern multi-agent systems ranging from sensor networks monitoring critical infrastructure to crowdsourcing platforms aggregating human intelligence can suffer significant performance degradation due to systematic biases that vary with environmental conditions. Current approaches either ignore these biases, leading to suboptimal decisions, or require expensive calibration procedures that are often infeasible in practice. This performance gap has real consequences: inaccurate environmental monitoring, unreliable financial predictions, and flawed aggregation of human judgments. This paper addresses the fundamental question: when can we learn and correct for these unknown biases to recover near-optimal performance, and when is such learning futile? We develop a theoretical framework that decomposes biases into learnable systematic components and irreducible stochastic components, introducing the concept of learnability ratio as the fraction of bias variance predictable from observable covariates. This ratio determines whether bias learning is worthwhile for a given system. We prove that the achievable performance improvement is fundamentally bounded by this learnability ratio, providing system designers with quantitative guidance on when to invest in bias learning versus simpler approaches. We present the Adaptive Bias Learning and Optimal Combining (ABLOC) algorithm, which iteratively learns bias-correcting transformations while optimizing combination weights through closedform solutions, guaranteeing convergence to these theoretical bounds. Experimental validation demonstrates that systems with high learnability ratios can recover significant performance (we achieved 40%-70% of theoretical maximum improvement in our examples), while those with low learnability show minimal benefit, validating our diagnostic criteria for practical deployment decisions.

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


### Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning
**Date:** 2025-10-27 | **Arxiv:** [2412.00661](https://arxiv.org/abs/2412.00661)

#### Abstract
Designing efficient algorithms for multi-agent reinforcement learning (MARL) is fundamentally challenging because the size of the joint state and action spaces grows exponentially in the number of agents. These difficulties are exacerbated when balancing sequential global decision-making with local agent interactions. In this work, we propose a new algorithm $\texttt{SUBSAMPLE-MFQ}$ ($\textbf{Subsample}$-$\textbf{M}$ean-$\textbf{F}$ield-$\textbf{Q}$-learning) and a decentralized randomized policy for a system with $n$ agents. For any $k\leq n$, our algorithm learns a policy for the system in time polynomial in $k$. We prove that this learned policy converges to the optimal policy on the order of $\tilde{O}(1/\sqrt{k})$ as the number of subsampled agents $k$ increases. In particular, this bound is independent of the number of agents $n$.

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


### Towards Principled Unsupervised Multi-Agent Reinforcement Learning
**Date:** 2025-10-21 | **Arxiv:** [2502.08365](https://arxiv.org/abs/2502.08365)

#### Abstract
In reinforcement learning, we typically refer to unsupervised pre-training when we aim to pre-train a policy without a priori access to the task specification, i.e. rewards, to be later employed for efficient learning of downstream tasks. In single-agent settings, the problem has been extensively studied and mostly understood. A popular approach, called task-agnostic exploration, casts the unsupervised objective as maximizing the entropy of the state distribution induced by the agent's policy, from which principles and methods follow.   In contrast, little is known about it in multi-agent settings, which are ubiquitous in the real world. What are the pros and cons of alternative problem formulations in this setting? How hard is the problem in theory, how can we solve it in practice? In this paper, we address these questions by first characterizing those alternative formulations and highlighting how the problem, even when tractable in theory, is non-trivial in practice. Then, we present a scalable, decentralized, trust-region policy search algorithm to address the problem in practical settings. Finally, we provide numerical validations to both corroborate the theoretical findings and pave the way for unsupervised multi-agent reinforcement learning via task-agnostic exploration in challenging domains, showing that optimizing for a specific objective, namely mixture entropy, provides an excellent trade-off between tractability and performances.

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


### A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning
**Date:** 2025-10-21 | **Arxiv:** [2510.17697](https://arxiv.org/abs/2510.17697)

#### Abstract
Steering cooperative multi-agent reinforcement learning (MARL) towards desired outcomes is challenging, particularly when the global guidance from a human on the whole multi-agent system is impractical in a large-scale MARL. On the other hand, designing external mechanisms (e.g., intrinsic rewards and human feedback) to coordinate agents mostly relies on empirical studies, lacking a easy-to-use research tool. In this work, we employ multi-agent influence diagrams (MAIDs) as a graphical framework to address the above issues. First, we introduce the concept of MARL interaction paradigms (orthogonal to MARL learning paradigms), using MAIDs to analyze and visualize both unguided self-organization and global guidance mechanisms in MARL. Then, we design a new MARL interaction paradigm, referred to as the targeted intervention paradigm that is applied to only a single targeted agent, so the problem of global guidance can be mitigated. In implementation, we introduce a causal inference technique, referred to as Pre-Strategy Intervention (PSI), to realize the targeted intervention paradigm. Since MAIDs can be regarded as a special class of causal diagrams, a composite desired outcome that integrates the primary task goal and an additional desired outcome can be achieved by maximizing the corresponding causal effect through the PSI. Moreover, the bundled relevance graph analysis of MAIDs provides a tool to identify whether an MARL learning paradigm is workable under the design of an MARL interaction paradigm. In experiments, we demonstrate the effectiveness of our proposed targeted intervention, and verify the result of relevance graph analysis.

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


### Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning
**Date:** 2025-10-16 | **Arxiv:** [2510.11824](https://arxiv.org/abs/2510.11824)

#### Abstract
In cooperative Multi-Agent Reinforcement Learning (MARL), it is a common practice to tune hyperparameters in ideal simulated environments to maximize cooperative performance. However, policies tuned for cooperation often fail to maintain robustness and resilience under real-world uncertainties. Building trustworthy MARL systems requires a deep understanding of robustness, which ensures stability under uncertainties, and resilience, the ability to recover from disruptions--a concept extensively studied in control systems but largely overlooked in MARL. In this paper, we present a large-scale empirical study comprising over 82,620 experiments to evaluate cooperation, robustness, and resilience in MARL across 4 real-world environments, 13 uncertainty types, and 15 hyperparameters. Our key findings are: (1) Under mild uncertainty, optimizing cooperation improves robustness and resilience, but this link weakens as perturbations intensify. Robustness and resilience also varies by algorithm and uncertainty type. (2) Robustness and resilience do not generalize across uncertainty modalities or agent scopes: policies robust to action noise for all agents may fail under observation noise on a single agent. (3) Hyperparameter tuning is critical for trustworthy MARL: surprisingly, standard practices like parameter sharing, GAE, and PopArt can hurt robustness, while early stopping, high critic learning rates, and Leaky ReLU consistently help. By optimizing hyperparameters only, we observe substantial improvement in cooperation, robustness and resilience across all MARL backbones, with the phenomenon also generalizing to robust MARL methods across these backbones. Code and results available at https://github.com/BUAA-TrustworthyMARL/adv_marl_benchmark .

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
* **Limits:** However, policies tuned for cooperation often fail to maintain robustness and resilience under real-world uncertainties.
* **Signal Tags:** #ai

---


### Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs
**Date:** 2025-10-15 | **Arxiv:** [2510.11062](https://arxiv.org/abs/2510.11062)

#### Abstract
Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models.   We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.

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
* **Limits:** However, applying on-policy RL to MAS remains underexplored and presents unique challenges.
* **Signal Tags:** #ai

---


### OrbitZoo: Multi-Agent Reinforcement Learning Environment for Orbital Dynamics
**Date:** 2025-10-15 | **Arxiv:** [2504.04160](https://arxiv.org/abs/2504.04160)

#### Abstract
The increasing number of satellites and orbital debris has made space congestion a critical issue, threatening satellite safety and sustainability. Challenges such as collision avoidance, station-keeping, and orbital maneuvering require advanced techniques to handle dynamic uncertainties and multi-agent interactions. Reinforcement learning (RL) has shown promise in this domain, enabling adaptive, autonomous policies for space operations; however, many existing RL frameworks rely on custom-built environments developed from scratch, which often use simplified models and require significant time to implement and validate the orbital dynamics, limiting their ability to fully capture real-world complexities. To address this, we introduce OrbitZoo, a versatile multi-agent RL environment built on a high-fidelity industry standard library, that enables realistic data generation, supports scenarios like collision avoidance and cooperative maneuvers, and ensures robust and accurate orbital dynamics. The environment is validated against a real satellite constellation, Starlink, achieving a Mean Absolute Percentage Error (MAPE) of 0.16% compared to real-world data. This validation ensures reliability for generating high-fidelity simulations and enabling autonomous and independent satellite operations.

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
* **Limits:** however, many existing RL frameworks rely on custom-built environments developed from scratch, which often use simplified models and require significant time to implement and validate the orbital dynamics, limiting their ability to fully capture real-world complexities.
* **Signal Tags:** #ai

---


### Climate Surrogates for Scalable Multi-Agent Reinforcement Learning: A Case Study with CICERO-SCM
**Date:** 2025-10-10 | **Arxiv:** [2510.07971](https://arxiv.org/abs/2510.07971)

#### Abstract
Climate policy studies require models that capture the combined effects of multiple greenhouse gases on global temperature, but these models are computationally expensive and difficult to embed in reinforcement learning. We present a multi-agent reinforcement learning (MARL) framework that integrates a high-fidelity, highly efficient climate surrogate directly in the environment loop, enabling regional agents to learn climate policies under multi-gas dynamics. As a proof of concept, we introduce a recurrent neural network architecture pretrained on ($20{,}000$) multi-gas emission pathways to surrogate the climate model CICERO-SCM. The surrogate model attains near-simulator accuracy with global-mean temperature RMSE $\approx 0.0004 \mathrm{K}$ and approximately $1000\times$ faster one-step inference. When substituted for the original simulator in a climate-policy MARL setting, it accelerates end-to-end training by $>\!100\times$. We show that the surrogate and simulator converge to the same optimal policies and propose a methodology to assess this property in cases where using the simulator is intractable. Our work allows to bypass the core computational bottleneck without sacrificing policy fidelity, enabling large-scale multi-agent experiments across alternative climate-policy regimes with multi-gas dynamics and high-fidelity climate response.

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


### Video Game Level Design as a Multi-Agent Reinforcement Learning Problem
**Date:** 2025-10-07 | **Arxiv:** [2510.04862](https://arxiv.org/abs/2510.04862)

#### Abstract
Procedural Content Generation via Reinforcement Learning (PCGRL) offers a method for training controllable level designer agents without the need for human datasets, using metrics that serve as proxies for level quality as rewards. Existing PCGRL research focuses on single generator agents, but are bottlenecked by the need to frequently recalculate heuristics of level quality and the agent's need to navigate around potentially large maps. By framing level generation as a multi-agent problem, we mitigate the efficiency bottleneck of single-agent PCGRL by reducing the number of reward calculations relative to the number of agent actions. We also find that multi-agent level generators are better able to generalize to out-of-distribution map shapes, which we argue is due to the generators' learning more local, modular design policies. We conclude that treating content generation as a distributed, multi-agent task is beneficial for generating functional artifacts at scale.

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


### LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science
**Date:** 2025-10-03 | **Arxiv:** [2510.01285](https://arxiv.org/abs/2510.01285)

#### Abstract
The rapid advancement of Large Language Models (LLMs) has opened new opportunities in data science, yet their practical deployment is often constrained by the challenge of discovering relevant data within large heterogeneous data lakes. Existing methods struggle with this: single-agent systems are quickly overwhelmed by large, heterogeneous files in the large data lakes, while multi-agent systems designed based on a master-slave paradigm depend on a rigid central controller for task allocation that requires precise knowledge of each sub-agent's capabilities. To address these limitations, we propose a novel multi-agent communication paradigm inspired by the blackboard architecture for traditional AI models. In this framework, a central agent posts requests to a shared blackboard, and autonomous subordinate agents -- either responsible for a partition of the data lake or general information retrieval -- volunteer to respond based on their capabilities. This design improves scalability and flexibility by eliminating the need for a central coordinator to have prior knowledge of all sub-agents' expertise. We evaluate our method on three benchmarks that require explicit data discovery: KramaBench and modified versions of DS-Bench and DA-Code to incorporate data discovery. Experimental results demonstrate that the blackboard architecture substantially outperforms baselines, including RAG and the master-slave multi-agent paradigm, achieving between 13% to 57% relative improvement in end-to-end task success and up to a 9% relative gain in F1 score for data discovery over the best-performing baselines across both proprietary and open-source LLMs. Our findings establish the blackboard paradigm as a scalable and generalizable communication framework for multi-agent systems.

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


### Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting
**Date:** 2025-10-02 | **Arxiv:** [2510.00401](https://arxiv.org/abs/2510.00401)

#### Abstract
Long-horizon motion forecasting for multiple autonomous robots is challenging due to non-linear agent interactions, compounding prediction errors, and continuous-time evolution of dynamics. Learned dynamics of such a system can be useful in various applications such as travel time prediction, prediction-guided planning and generative simulation. In this work, we aim to develop an efficient trajectory forecasting model conditioned on multi-agent goals. Motivated by the recent success of physics-guided deep learning for partially known dynamical systems, we develop a model based on neural Controlled Differential Equations (CDEs) for long-horizon motion forecasting. Unlike discrete-time methods such as RNNs and transformers, neural CDEs operate in continuous time, allowing us to combine physics-informed constraints and biases to jointly model multi-robot dynamics. Our approach, named PINCoDE (Physics-Informed Neural Controlled Differential Equations), learns differential equation parameters that can be used to predict the trajectories of a multi-agent system starting from an initial condition. PINCoDE is conditioned on future goals and enforces physics constraints for robot motion over extended periods of time. We adopt a strategy that scales our model from 10 robots to 100 robots without the need for additional model parameters, while producing predictions with an average ADE below 0.5 m for a 1-minute horizon. Furthermore, progressive training with curriculum learning for our PINCoDE model results in a 2.7X reduction of forecasted pose error over 4 minute horizons compared to analytical models.

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


### Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives
**Date:** 2025-09-29 | **Arxiv:** [2509.22596](https://arxiv.org/abs/2509.22596)

#### Abstract
In this paper, we present two effective policy learning algorithms for multi-agent online coordination(MA-OC) problem. The first one, \texttt{MA-SPL}, not only can achieve the optimal $(1-\frac{c}{e})$-approximation guarantee for the MA-OC problem with submodular objectives but also can handle the unexplored $α$-weakly DR-submodular and $(γ,β)$-weakly submodular scenarios, where $c$ is the curvature of the investigated submodular functions, $α$ denotes the diminishing-return(DR) ratio and the tuple $(γ,β)$ represents the submodularity ratios. Subsequently, in order to reduce the reliance on the unknown parameters $α,γ,β$ inherent in the \texttt{MA-SPL} algorithm, we further introduce the second online algorithm named \texttt{MA-MPL}. This \texttt{MA-MPL} algorithm is entirely \emph{parameter-free} and simultaneously can maintain the same approximation ratio as the first \texttt{MA-SPL} algorithm. The core of our \texttt{MA-SPL} and \texttt{MA-MPL} algorithms is a novel continuous-relaxation technique termed as \emph{policy-based continuous extension}. Compared with the well-established \emph{multi-linear extension}, a notable advantage of this new \emph{policy-based continuous extension} is its ability to provide a lossless rounding scheme for any set function, thereby enabling us to tackle the challenging weakly submodular objectives. Finally, extensive simulations are conducted to validate the effectiveness of our proposed algorithms.

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


### LEED: A Highly Efficient and Scalable LLM-Empowered Expert Demonstrations Framework for Multi-Agent Reinforcement Learning
**Date:** 2025-09-19 | **Arxiv:** [2509.14680](https://arxiv.org/abs/2509.14680)

#### Abstract
Multi-agent reinforcement learning (MARL) holds substantial promise for intelligent decision-making in complex environments. However, it suffers from a coordination and scalability bottleneck as the number of agents increases. To address these issues, we propose the LLM-empowered expert demonstrations framework for multi-agent reinforcement learning (LEED). LEED consists of two components: a demonstration generation (DG) module and a policy optimization (PO) module. Specifically, the DG module leverages large language models to generate instructions for interacting with the environment, thereby producing high-quality demonstrations. The PO module adopts a decentralized training paradigm, where each agent utilizes the generated demonstrations to construct an expert policy loss, which is then integrated with its own policy loss. This enables each agent to effectively personalize and optimize its local policy based on both expert knowledge and individual experience. Experimental results show that LEED achieves superior sample efficiency, time efficiency, and robust scalability compared to state-of-the-art baselines.

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
* **Limits:** However, it suffers from a coordination and scalability bottleneck as the number of agents increases.
* **Signal Tags:** #ai

---


### SafeDiver: Cooperative AUV-USV Assisted Diver Communication via Multi-agent Reinforcement Learning Approach
**Date:** 2025-09-16 | **Arxiv:** [2509.11508](https://arxiv.org/abs/2509.11508)

#### Abstract
As underwater human activities are increasing, the demand for underwater communication service presents a significant challenge. Existing underwater diver communication methods face hurdles due to inherent disadvantages and complex underwater environments. To address this issue, we propose a scheme that utilizes maritime unmanned systems to assist divers with reliable and high-speed communication. Multiple AUVs are equipped with optical and acoustic multimodal communication devices as relay nodes, providing adaptive communication services based on changes in the diver's activity area. By using a multi-agent reinforcement learning (MARL) approach to control the cooperative movement of AUVs, high-speed and reliable data transmission between divers can be achieved. At the same time, utilizing the advantages of on-demand deployment and wide coverage of unmanned surface vehicles (USVs) as surface relay nodes to coordinate and forward information from AUVs, and controlling AUVs to adaptively select relay USV nodes for data transmission, high-quality communication between divers and surface platform can be achieved. Through simulation verification, the proposed scheme can effectively achieve reliable and high-speed communication for divers.

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


### $K$-Level Policy Gradients for Multi-Agent Reinforcement Learning
**Date:** 2025-09-16 | **Arxiv:** [2509.12117](https://arxiv.org/abs/2509.12117)

#### Abstract
Actor-critic algorithms for deep multi-agent reinforcement learning (MARL) typically employ a policy update that responds to the current strategies of other agents. While being straightforward, this approach does not account for the updates of other agents at the same update step, resulting in miscoordination. In this paper, we introduce the $K$-Level Policy Gradient (KPG), a method that recursively updates each agent against the updated policies of other agents, speeding up the discovery of effective coordinated policies. We theoretically prove that KPG with finite iterates achieves monotonic convergence to a local Nash equilibrium under certain conditions. We provide principled implementations of KPG by applying it to the deep MARL algorithms MAPPO, MADDPG, and FACMAC. Empirically, we demonstrate superior performance over existing deep MARL algorithms in StarCraft II and multi-agent MuJoCo.

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


### VariAntNet: Learning Decentralized Control of Multi-Agent Systems
**Date:** 2025-09-03 | **Arxiv:** [2509.02271](https://arxiv.org/abs/2509.02271)

#### Abstract
A simple multi-agent system can be effectively utilized in disaster response applications, such as firefighting. Such a swarm is required to operate in complex environments with limited local sensing and no reliable inter-agent communication or centralized control. These simple robotic agents, also known as Ant Robots, are defined as anonymous agents that possess limited sensing capabilities, lack a shared coordinate system, and do not communicate explicitly with one another. A key challenge for simple swarms lies in maintaining cohesion and avoiding fragmentation despite limited-range sensing. Recent advances in machine learning offer effective solutions to some of the classical decentralized control challenges. We propose VariAntNet, a deep learning-based decentralized control model designed to facilitate agent swarming and collaborative task execution. VariAntNet includes geometric features extraction from unordered, variable-sized local observations. It incorporates a neural network architecture trained with a novel, differentiable, multi-objective, mathematically justified loss function that promotes swarm cohesiveness by utilizing the properties of the visibility graph Laplacian matrix. VariAntNet is demonstrated on the fundamental multi-agent gathering task, where agents with bearing-only and limited-range sensing must gather at some location. VariAntNet significantly outperforms an existing analytical solution, achieving more than double the convergence rate while maintaining high swarm connectivity across varying swarm sizes. While the analytical solution guarantees cohesion, it is often too slow in practice. In time-critical scenarios, such as emergency response operations where lives are at risk, slower analytical methods are impractical and justify the loss of some agents within the swarm. This paper presents and analyzes this trade-off in detail.

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


### Probabilistic Multi-Agent Aircraft Landing Time Prediction
**Date:** 2025-12-10 | **Arxiv:** [2512.08281](https://arxiv.org/abs/2512.08281)

#### Abstract
Accurate and reliable aircraft landing time prediction is essential for effective resource allocation in air traffic management. However, the inherent uncertainty of aircraft trajectories and traffic flows poses significant challenges to both prediction accuracy and trustworthiness. Therefore, prediction models should not only provide point estimates of aircraft landing times but also the uncertainties associated with these predictions. Furthermore, aircraft trajectories are frequently influenced by the presence of nearby aircraft through air traffic control interventions such as radar vectoring. Consequently, landing time prediction models must account for multi-agent interactions in the airspace. In this work, we propose a probabilistic multi-agent aircraft landing time prediction framework that provides the landing times of multiple aircraft as distributions. We evaluate the proposed framework using an air traffic surveillance dataset collected from the terminal airspace of the Incheon International Airport in South Korea. The results demonstrate that the proposed model achieves higher prediction accuracy than the baselines and quantifies the associated uncertainties of its outcomes. In addition, the model uncovered underlying patterns in air traffic control through its attention scores, thereby enhancing explainability.

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
* **Limits:** However, the inherent uncertainty of aircraft trajectories and traffic flows poses significant challenges to both prediction accuracy and trustworthiness.
* **Signal Tags:** #ai

---


### Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection
**Date:** 2025-11-19 | **Arxiv:** [2511.13759](https://arxiv.org/abs/2511.13759)

#### Abstract
Accurate detection of offensive content on social media demands high-quality labeled data; however, such data is often scarce due to the low prevalence of offensive instances and the high cost of manual annotation. To address this low-resource challenge, we propose a self-training framework that leverages abundant unlabeled data through collaborative pseudo-labeling. Starting with a lightweight classifier trained on limited labeled data, our method iteratively assigns pseudo-labels to unlabeled instances with the support of Multi-Agent Vision-Language Models (MA-VLMs). Un-labeled data on which the classifier and MA-VLMs agree are designated as the Agreed-Unknown set, while conflicting samples form the Disagreed-Unknown set. To enhance label reliability, MA-VLMs simulate dual perspectives, moderator and user, capturing both regulatory and subjective viewpoints. The classifier is optimized using a novel Positive-Negative-Unlabeled (PNU) loss, which jointly exploits labeled, Agreed-Unknown, and Disagreed-Unknown data while mitigating pseudo-label noise. Experiments on benchmark datasets demonstrate that our framework substantially outperforms baselines under limited supervision and approaches the performance of large-scale models

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
* **Limits:** however, such data is often scarce due to the low prevalence of offensive instances and the high cost of manual annotation.
* **Signal Tags:** #ai

---


### Chicken Swarm Kernel Particle Filter: A Structured Rejuvenation Approach with KLD-Efficient Sampling
**Date:** 2025-11-18 | **Arxiv:** [2511.12222](https://arxiv.org/abs/2511.12222)

#### Abstract
Particle filters (PFs) are often combined with swarm intelligence (SI) algorithms, such as Chicken Swarm Optimization (CSO), for particle rejuvenation. Separately, Kullback--Leibler divergence (KLD) sampling is a common strategy for adaptively sizing the particle set. However, the theoretical interaction between SI-based rejuvenation kernels and KLD-based adaptive sampling is not yet fully understood.   This paper investigates this specific interaction. We analyze, under a simplified modeling framework, the effect of the CSO rejuvenation step on the particle set distribution. We propose that the fitness-driven updates inherent in CSO can be approximated as a form of mean-square contraction. This contraction tends to produce a particle distribution that is more concentrated than that of a baseline PF, or in mathematical terms, a distribution that is plausibly more ``peaked'' in a majorization sense.   By applying Karamata's inequality to the concave function that governs the expected bin occupancy in KLD-sampling, our analysis suggests a connection: under the stated assumptions, the CSO-enhanced PF (CPF) is expected to require a lower \emph{expected} particle count than the standard PF to satisfy the same statistical error bound. The goal of this study is not to provide a fully general proof, but rather to offer a tractable theoretical framework that helps to interpret the computational efficiency empirically observed when combining these techniques, and to provide a starting point for designing more efficient adaptive filters.

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
* **Limits:** However, the theoretical interaction between SI-based rejuvenation kernels and KLD-based adaptive sampling is not yet fully understood.
* **Signal Tags:** #ai

---


### Multi-agent Markov Entanglement
**Date:** 2025-11-14 | **Arxiv:** [2506.02385](https://arxiv.org/abs/2506.02385)

#### Abstract
Value decomposition has long been a fundamental technique in multi-agent dynamic programming and reinforcement learning (RL). Specifically, the value function of a global state $(s_1,s_2,\ldots,s_N)$ is often approximated as the sum of local functions: $V(s_1,s_2,\ldots,s_N)\approx\sum_{i=1}^N V_i(s_i)$. This approach traces back to the index policy in restless multi-armed bandit problems and has found various applications in modern RL systems. However, the theoretical justification for why this decomposition works so effectively remains underexplored.   In this paper, we uncover the underlying mathematical structure that enables value decomposition. We demonstrate that a multi-agent Markov decision process (MDP) permits value decomposition if and only if its transition matrix is not "entangled" -- a concept analogous to quantum entanglement in quantum physics. Drawing inspiration from how physicists measure quantum entanglement, we introduce how to measure the "Markov entanglement" for multi-agent MDPs and show that this measure can be used to bound the decomposition error in general multi-agent MDPs.   Using the concept of Markov entanglement, we proved that a widely-used class of index policies is weakly entangled and enjoys a sublinear $\mathcal O(\sqrt{N})$ scale of decomposition error for $N$-agent systems. Finally, we show how Markov entanglement can be efficiently estimated in practice, providing practitioners with an empirical proxy for the quality of value decomposition.

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
* **Limits:** However, the theoretical justification for why this decomposition works so effectively remains underexplored.
* **Signal Tags:** #ai

---


### ARAC: Adaptive Regularized Multi-Agent Soft Actor-Critic in Graph-Structured Adversarial Games
**Date:** 2025-11-12 | **Arxiv:** [2511.08412](https://arxiv.org/abs/2511.08412)

#### Abstract
In graph-structured multi-agent reinforcement learning (MARL) adversarial tasks such as pursuit and confrontation, agents must coordinate under highly dynamic interactions, where sparse rewards hinder efficient policy learning. We propose Adaptive Regularized Multi-Agent Soft Actor-Critic (ARAC), which integrates an attention-based graph neural network (GNN) for modeling agent dependencies with an adaptive divergence regularization mechanism. The GNN enables expressive representation of spatial relations and state features in graph environments. Divergence regularization can serve as policy guidance to alleviate the sparse reward problem, but it may lead to suboptimal convergence when the reference policy itself is imperfect. The adaptive divergence regularization mechanism enables the framework to exploit reference policies for efficient exploration in the early stages, while gradually reducing reliance on them as training progresses to avoid inheriting their limitations. Experiments in pursuit and confrontation scenarios demonstrate that ARAC achieves faster convergence, higher final success rates, and stronger scalability across varying numbers of agents compared with MARL baselines, highlighting its effectiveness in complex graph-structured environments.

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


### Multi-agent Coordination via Flow Matching
**Date:** 2025-11-10 | **Arxiv:** [2511.05005](https://arxiv.org/abs/2511.05005)

#### Abstract
This work presents MAC-Flow, a simple yet expressive framework for multi-agent coordination. We argue that requirements of effective coordination are twofold: (i) a rich representation of the diverse joint behaviors present in offline data and (ii) the ability to act efficiently in real time. However, prior approaches often sacrifice one for the other, i.e., denoising diffusion-based solutions capture complex coordination but are computationally slow, while Gaussian policy-based solutions are fast but brittle in handling multi-agent interaction. MAC-Flow addresses this trade-off by first learning a flow-based representation of joint behaviors, and then distilling it into decentralized one-step policies that preserve coordination while enabling fast execution. Across four different benchmarks, including $12$ environments and $34$ datasets, MAC-Flow alleviates the trade-off between performance and computational cost, specifically achieving about $\boldsymbol{\times14.5}$ faster inference compared to diffusion-based MARL methods, while maintaining good performance. At the same time, its inference speed is similar to that of prior Gaussian policy-based offline multi-agent reinforcement learning (MARL) methods.

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
* **Limits:** However, prior approaches often sacrifice one for the other, i.
* **Signal Tags:** #ai

---


### Scaling Multi-Agent Environment Co-Design with Diffusion Models
**Date:** 2025-11-06 | **Arxiv:** [2511.03100](https://arxiv.org/abs/2511.03100)

#### Abstract
The agent-environment co-design paradigm jointly optimises agent policies and environment configurations in search of improved system performance. With application domains ranging from warehouse logistics to windfarm management, co-design promises to fundamentally change how we deploy multi-agent systems. However, current co-design methods struggle to scale. They collapse under high-dimensional environment design spaces and suffer from sample inefficiency when addressing moving targets inherent to joint optimisation. We address these challenges by developing Diffusion Co-Design (DiCoDe), a scalable and sample-efficient co-design framework pushing co-design towards practically relevant settings. DiCoDe incorporates two core innovations. First, we introduce Projected Universal Guidance (PUG), a sampling technique that enables DiCoDe to explore a distribution of reward-maximising environments while satisfying hard constraints such as spatial separation between obstacles. Second, we devise a critic distillation mechanism to share knowledge from the reinforcement learning critic, ensuring that the guided diffusion model adapts to evolving agent policies using a dense and up-to-date learning signal. Together, these improvements lead to superior environment-policy pairs when validated on challenging multi-agent environment co-design benchmarks including warehouse automation, multi-agent pathfinding and wind farm optimisation. Our method consistently exceeds the state-of-the-art, achieving, for example, 39% higher rewards in the warehouse setting with 66% fewer simulation samples. This sets a new standard in agent-environment co-design, and is a stepping stone towards reaping the rewards of co-design in real world domains.

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
* **Limits:** However, current co-design methods struggle to scale.
* **Signal Tags:** #ai

---
