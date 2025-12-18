# Vol 19 LLM Agents   Tool Use
*Enriched by BITCOREOS | Phase 4 Batch 4*

---

### Introducing agent-to-agent protocol support in Amazon Bedrock AgentCore Runtime
**Date:** 2025-11-11 | **Arxiv:** [](https://arxiv.org/abs/)

#### Abstract
We recently announced the support for Agent-to-Agent (A2A) protocol on Amazon Bedrock AgentCore Runtime. With this addition, agents can discover peers, share capabilities, and coordinate actions across platforms using standardized communication.  Thi...

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


### Orion: A Unified Visual Agent for Multimodal Perception, Advanced Visual Reasoning and Execution
**Date:** 2025-11-19 | **Arxiv:** [2511.14210](https://arxiv.org/abs/2511.14210)

#### Abstract
We introduce Orion, a visual agent that integrates vision-based reasoning with tool-augmented execution to achieve powerful, precise, multi-step visual intelligence across images, video, and documents. Unlike traditional vision-language models that generate descriptive outputs, Orion orchestrates a suite of specialized computer vision tools, including object detection, keypoint localization, panoptic segmentation, Optical Character Recognition (OCR), and geometric analysis, to execute complex multi-step visual workflows. The system achieves competitive performance across MMMU, MMBench, DocVQA, and MMLongBench while extending monolithic VLM capabilities to production-grade visual intelligence. Through its agentic, tool-augmented approach, Orion enables autonomous visual reasoning that bridges neural perception with symbolic execution, marking the transition from passive visual understanding to active, tool-driven visual intelligence.   Try Orion for free at: https://chat.vlm.run   Learn more at: https://www.vlm.run/orion

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


### Socrates-Mol: Self-Oriented Cognitive Reasoning through Autonomous Trial-and-Error with Empirical-Bayesian Screening for Molecules
**Date:** 2025-11-18 | **Arxiv:** [2511.11769](https://arxiv.org/abs/2511.11769)

#### Abstract
Molecular property prediction is fundamental to chemical engineering applications such as solvent screening. We present Socrates-Mol, a framework that transforms language models into empirical Bayesian reasoners through context engineering, addressing cold start problems without model fine-tuning. The system implements a reflective-prediction cycle where initial outputs serve as priors, retrieved molecular cases provide evidence, and refined predictions form posteriors, extracting reusable chemical rules from sparse data. We introduce ranking tasks aligned with industrial screening priorities and employ cross-model self-consistency across five language models to reduce variance. Experiments on amine solvent LogP prediction reveal task-dependent patterns: regression achieves 72% MAE reduction and 112% R-squared improvement through self-consistency, while ranking tasks show limited gains due to systematic multi-model biases. The framework reduces deployment costs by over 70% compared to full fine-tuning, providing a scalable solution for molecular property prediction while elucidating the task-adaptive nature of self-consistency mechanisms.

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
* **Construct:** Formal Reasoning
* **Layer:** Application
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Build a device management agent with Amazon Bedrock AgentCore
**Date:** 2025-10-14 | **Arxiv:** [](https://arxiv.org/abs/)

#### Abstract
The proliferation of Internet of Things (IoT) devices has transformed how we interact with our environments, from homes to industrial settings. However, as the number of connected devices grows, so does the complexity of managing them. Traditional de...

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
* **Limits:** However, as the number of connected devices grows, so does the complexity of managing them.
* **Signal Tags:** #ai

---


### Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything
**Date:** 2025-11-05 | **Arxiv:** [2511.02834](https://arxiv.org/abs/2511.02834)

#### Abstract
Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available.

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


### Surrogate modeling of Cellular-Potts Agent-Based Models as a segmentation task using the U-Net neural network architecture
**Date:** 2025-11-05 | **Arxiv:** [2505.00316](https://arxiv.org/abs/2505.00316)

#### Abstract
The Cellular-Potts model is a powerful and ubiquitous framework for developing computational models for simulating complex multicellular biological systems. Cellular-Potts models (CPMs) are often computationally expensive due to the explicit modeling of interactions among large numbers of individual model agents and diffusive fields described by partial differential equations (PDEs). In this work, we develop a convolutional neural network (CNN) surrogate model using a U-Net architecture that accounts for periodic boundary conditions. We use this model to accelerate the evaluation of a mechanistic CPM previously used to investigate in vitro vasculogenesis. The surrogate model was trained to predict 100 computational steps ahead (Monte-Carlo steps, MCS), accelerating simulation evaluations by a factor of 590 times compared to CPM code execution. Over multiple recursive evaluations, our model effectively captures the emergent behaviors demonstrated by the original Cellular-Potts model of such as vessel sprouting, extension and anastomosis, and contraction of vascular lacunae. This approach demonstrates the potential for deep learning to serve as efficient surrogate models for CPM simulations, enabling faster evaluation of computationally expensive CPM of biological processes at greater spatial and temporal scales.

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


### SimuRA: A World-Model-Driven Simulative Reasoning Architecture for General Goal-Oriented Agents
**Date:** 2025-10-27 | **Arxiv:** [2507.23773](https://arxiv.org/abs/2507.23773)

#### Abstract
AI agents built on foundation models hold enormous promise. Current practice, however, focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also faces practical limitations from black-box autoregressive reasoning, where decisions unfold token by token without explicit simulation or counterfactual evaluation of outcomes. Humans, on the other hand, reason and plan by mentally simulating the consequences of actions within an internal model of the world -- a capability that supports flexible, goal-directed behavior across diverse contexts. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of an optimal agent in any general environment, SimuRA addresses the limitations of black-box autoregressive reasoning by incorporating the world model for planning via simulation. Our prototype world model is implemented using LLMs as a substrate, leveraging the natural language as a discrete, hierarchical representation grounded in concepts for planning, while remaining model-agnostic. On complex web-browsing tasks such as flight search, SimuRA improves the success rate from 0% to 32.2% compared to a representative open-web agent baseline. Across tasks, world-model-based planning achieves up to 124% higher task completion rates than a matched black-box autoregressive baseline, demonstrating the advantages of simulative reasoning. We release ReasonerAgent-Web, a web-browsing agent built on SimuRA, as an open-source research demo.

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
* **Limits:** however, focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also faces practical limitations from black-box autoregressive reasoning, where decisions unfold token by token without explicit simulation or counterfactual evaluation of outcomes.
* **Signal Tags:** #ai

---


### EchoAgent: Guideline-Centric Reasoning Agent for Echocardiography Measurement and Interpretation
**Date:** 2025-11-19 | **Arxiv:** [2511.13948](https://arxiv.org/abs/2511.13948)

#### Abstract
Purpose: Echocardiographic interpretation requires video-level reasoning and guideline-based measurement analysis, which current deep learning models for cardiac ultrasound do not support. We present EchoAgent, a framework that enables structured, interpretable automation for this domain. Methods: EchoAgent orchestrates specialized vision tools under Large Language Model (LLM) control to perform temporal localization, spatial measurement, and clinical interpretation. A key contribution is a measurement-feasibility prediction model that determines whether anatomical structures are reliably measurable in each frame, enabling autonomous tool selection. We curated a benchmark of diverse, clinically validated video-query pairs for evaluation. Results: EchoAgent achieves accurate, interpretable results despite added complexity of spatiotemporal video analysis. Outputs are grounded in visual evidence and clinical guidelines, supporting transparency and traceability. Conclusion: This work demonstrates the feasibility of agentic, guideline-aligned reasoning for echocardiographic video analysis, enabled by task-specific tools and full video-level automation. EchoAgent sets a new direction for trustworthy AI in cardiac ultrasound.

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


### Mapping fNIRS Signals to Agent Performance: Toward Reinforcement Learning from Neural Feedback
**Date:** 2025-11-18 | **Arxiv:** [2511.12844](https://arxiv.org/abs/2511.12844)

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) is a methodology that aligns agent behavior with human preferences by integrating human feedback into the agent's training process. We introduce a possible framework that employs passive Brain-Computer Interfaces (BCI) to guide agent training from implicit neural signals. We present and release a novel dataset of functional near-infrared spectroscopy (fNIRS) recordings collected from 25 human participants across three domains: a Pick-and-Place Robot, Lunar Lander, and Flappy Bird. We train classifiers to predict levels of agent performance (optimal, sub-optimal, or worst-case) from windows of preprocessed fNIRS feature vectors, achieving an average F1 score of 67% for binary classification and 46% for multi-class models averaged across conditions and domains. We also train regressors to predict the degree of deviation between an agent's chosen action and a set of near-optimal policies, providing a continuous measure of performance. We evaluate cross-subject generalization and demonstrate that fine-tuning pre-trained models with a small sample of subject-specific data increases average F1 scores by 17% and 41% for binary and multi-class models, respectively. Our work demonstrates that mapping implicit fNIRS signals to agent performance is feasible and can be improved, laying the foundation for future brain-driven RLHF systems.

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


### Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
**Date:** 2025-11-04 | **Arxiv:** [2511.00088](https://arxiv.org/abs/2511.00088)

#### Abstract
End-to-end architectures trained via imitation learning have advanced autonomous driving by scaling model size and data, yet performance remains brittle in safety-critical long-tail scenarios where supervision is sparse and causal understanding is limited. To address this, we introduce Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning to enhance decision-making in complex driving scenarios. Our approach features three key innovations: (1) the Chain of Causation (CoC) dataset, built through a hybrid auto-labeling and human-in-the-loop pipeline producing decision-grounded, causally linked reasoning traces aligned with driving behaviors; (2) a modular VLA architecture combining Cosmos-Reason, a Vision-Language Model pre-trained for Physical AI applications, with a diffusion-based trajectory decoder that generates dynamically feasible plans in real time; (3) a multi-stage training strategy using supervised fine-tuning to elicit reasoning and reinforcement learning (RL) to optimize reasoning quality via large reasoning model feedback and enforce reasoning-action consistency. Evaluation shows AR1 achieves up to a 12% improvement in planning accuracy on challenging cases compared to a trajectory-only baseline, with a 35% reduction in off-road rate and 25% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% as measured by a large reasoning model critic and reasoning-action consistency by 37%. Model scaling from 0.5B to 7B parameters shows consistent improvements. On-vehicle road tests confirm real-time performance (99 ms latency) and successful urban deployment. By bridging interpretable reasoning with precise control, AR1 demonstrates a practical path towards Level 4 autonomous driving. We plan to release AR1 models and a subset of the CoC in a future update.

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


### Beyond Prompt Engineering: Neuro-Symbolic-Causal Architecture for Robust Multi-Objective AI Agents
**Date:** 2025-10-29 | **Arxiv:** [2510.23682](https://arxiv.org/abs/2510.23682)

#### Abstract
Large language models show promise as autonomous decision-making agents, yet their deployment in high-stakes domains remains fraught with risk. Without architectural safeguards, LLM agents exhibit catastrophic brittleness: identical capabilities produce wildly different outcomes depending solely on prompt framing. We present Chimera, a neuro-symbolic-causal architecture that integrates three complementary components - an LLM strategist, a formally verified symbolic constraint engine, and a causal inference module for counterfactual reasoning. We benchmark Chimera against baseline architectures (LLM-only, LLM with symbolic constraints) across 52-week simulations in a realistic e-commerce environment featuring price elasticity, trust dynamics, and seasonal demand. Under organizational biases toward either volume or margin optimization, LLM-only agents fail catastrophically (total loss of \$99K in volume scenarios) or destroy brand trust (-48.6% in margin scenarios). Adding symbolic constraints prevents disasters but achieves only 43-87% of Chimera's profit. Chimera consistently delivers the highest returns (\$1.52M and \$1.96M respectively, some cases +\$2.2M) while improving brand trust (+1.8% and +10.8%, some cases +20.86%), demonstrating prompt-agnostic robustness. Our TLA+ formal verification proves zero constraint violations across all scenarios. These results establish that architectural design not prompt engineering determines the reliability of autonomous agents in production environments. We provide open-source implementations and interactive demonstrations for reproducibility.

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


### CARL: Critical Action Focused Reinforcement Learning for Multi-Step Agent
**Date:** 2025-12-05 | **Arxiv:** [2512.04949](https://arxiv.org/abs/2512.04949)

#### Abstract
Agents capable of accomplishing complex tasks through multiple interactions with the environment have emerged as a popular research direction. However, in such multi-step settings, the conventional group-level policy optimization algorithm becomes suboptimal because of its underlying assumption that each action holds equal contribution, which deviates significantly from reality. Our analysis reveals that only a small fraction of actions are critical in determining the final outcome. Building on this insight, we propose CARL, a critical-action-focused reinforcement learning algorithm tailored for multi-step agents. CARL achieves focused training through providing action-level optimization signals for high-criticality actions while excluding low-criticality actions from model update. Extensive experiments demonstrate that CARL achieves both stronger performance and higher efficiency during training and inference across diverse evaluation settings.

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
* **Limits:** However, in such multi-step settings, the conventional group-level policy optimization algorithm becomes suboptimal because of its underlying assumption that each action holds equal contribution, which deviates significantly from reality.
* **Signal Tags:** #ai

---


### Reducing Latency of LLM Search Agent via Speculation-based Algorithm-System Co-Design
**Date:** 2025-11-26 | **Arxiv:** [2511.20048](https://arxiv.org/abs/2511.20048)

#### Abstract
LLM-based search agents achieve strong performance but suffer from severe latency, as each step requires serialized LLM reasoning followed by action of tool execution. We revisit this bottleneck through the lens of speculation. While traditional predict-verify speculation paradigm can break serial execution, its benefit remains limited, as it retains the full original workload and adds extra inference overhead. We observe that early agent steps often involve simple evidence-gathering, where correct actions can often be predicted without full reasoning. Building on these observations, we present SPAgent, an algorithm-system co-design framework that expands the role of speculation in search agents to reduce latency. Algorithmically, SPAgent introduces a two-phase adaptive speculation mechanism that selectively omits verification when safe. System-wise, a two-level scheduler regulates speculative requests based on engine load to ensure speculation remains beneficial. We implement SPAgent in real-world systems. Across extensive experimental settings, SPAgent achieves up to $1.65\times$ end-to-end speedup while maintaining same or even achieving higher accuracy, enabling practical deployment of multi-step search agents.

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


### AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning
**Date:** 2025-11-25 | **Arxiv:** [2511.19304](https://arxiv.org/abs/2511.19304)

#### Abstract
Humans naturally adapt to diverse environments by learning underlying rules across worlds with different dynamics, observations, and reward structures. In contrast, existing agents typically demonstrate improvements via self-evolving within a single domain, implicitly assuming a fixed environment distribution. Cross-environment learning has remained largely unmeasured: there is no standard collection of controllable, heterogeneous environments, nor a unified way to represent how agents learn. We address these gaps in two steps. First, we propose AutoEnv, an automated framework that treats environments as factorizable distributions over transitions, observations, and rewards, enabling low-cost (4.12 USD on average) generation of heterogeneous worlds. Using AutoEnv, we construct AutoEnv-36, a dataset of 36 environments with 358 validated levels, on which seven language models achieve 12-49% normalized reward, demonstrating the challenge of AutoEnv-36. Second, we formalize agent learning as a component-centric process driven by three stages of Selection, Optimization, and Evaluation applied to an improvable agent component. Using this formulation, we design eight learning methods and evaluate them on AutoEnv-36. Empirically, the gain of any single learning method quickly decrease as the number of environments increases, revealing that fixed learning methods do not scale across heterogeneous environments. Environment-adaptive selection of learning methods substantially improves performance but exhibits diminishing returns as the method space expands. These results highlight both the necessity and the current limitations of agent learning for scalable cross-environment generalization, and position AutoEnv and AutoEnv-36 as a testbed for studying cross-environment agent learning. The code is avaiable at https://github.com/FoundationAgents/AutoEnv.

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


### Task Specific Sharpness Aware O-RAN Resource Management using Multi Agent Reinforcement Learning
**Date:** 2025-11-20 | **Arxiv:** [2511.15002](https://arxiv.org/abs/2511.15002)

#### Abstract
Next-generation networks utilize the Open Radio Access Network (O-RAN) architecture to enable dynamic resource management, facilitated by the RAN Intelligent Controller (RIC). While deep reinforcement learning (DRL) models show promise in optimizing network resources, they often struggle with robustness and generalizability in dynamic environments. This paper introduces a novel resource management approach that enhances the Soft Actor Critic (SAC) algorithm with Sharpness-Aware Minimization (SAM) in a distributed Multi-Agent RL (MARL) framework. Our method introduces an adaptive and selective SAM mechanism, where regularization is explicitly driven by temporal-difference (TD)-error variance, ensuring that only agents facing high environmental complexity are regularized. This targeted strategy reduces unnecessary overhead, improves training stability, and enhances generalization without sacrificing learning efficiency. We further incorporate a dynamic $ρ$ scheduling scheme to refine the exploration-exploitation trade-off across agents. Experimental results show our method significantly outperforms conventional DRL approaches, yielding up to a $22\%$ improvement in resource allocation efficiency and ensuring superior QoS satisfaction across diverse O-RAN slices.

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


### AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery
**Date:** 2025-11-17 | **Arxiv:** [2511.11257](https://arxiv.org/abs/2511.11257)

#### Abstract
The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery.

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


### Optimizing Electric Vehicle Charging Station Placement Using Reinforcement Learning and Agent-Based Simulations
**Date:** 2025-11-04 | **Arxiv:** [2511.01218](https://arxiv.org/abs/2511.01218)

#### Abstract
The rapid growth of electric vehicles (EVs) necessitates the strategic placement of charging stations to optimize resource utilization and minimize user inconvenience. Reinforcement learning (RL) offers an innovative approach to identifying optimal charging station locations; however, existing methods face challenges due to their deterministic reward systems, which limit efficiency. Because real-world conditions are dynamic and uncertain, a deterministic reward structure cannot fully capture the complexities of charging station placement. As a result, evaluation becomes costly and time-consuming, and less reflective of real-world scenarios. To address this challenge, we propose a novel framework that integrates deep RL with agent-based simulations to model EV movement and estimate charging demand in real time. Our approach employs a hybrid RL agent with dual Q-networks to select optimal locations and configure charging ports, guided by a hybrid reward function that combines deterministic factors with simulation-derived feedback. Case studies in Hanoi, Vietnam, show that our method reduces average waiting times by 53.28% compared to the initial state, outperforming static baseline methods. This scalable and adaptive solution enhances EV infrastructure planning, effectively addressing real-world complexities and improving user experience.

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
* **Limits:** however, existing methods face challenges due to their deterministic reward systems, which limit efficiency.
* **Signal Tags:** #ai

---


### Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL
**Date:** 2025-10-31 | **Arxiv:** [2505.22151](https://arxiv.org/abs/2505.22151)

#### Abstract
A key challenge in offline multi-agent reinforcement learning (MARL) is achieving effective many-agent multi-step coordination in complex environments. In this work, we propose Oryx, a novel algorithm for offline cooperative MARL to directly address this challenge. Oryx adapts the recently proposed retention-based architecture Sable and combines it with a sequential form of implicit constraint Q-learning (ICQ), to develop a novel offline autoregressive policy update scheme. This allows Oryx to solve complex coordination challenges while maintaining temporal coherence over long trajectories. We evaluate Oryx across a diverse set of benchmarks from prior works -- SMAC, RWARE, and Multi-Agent MuJoCo -- covering tasks of both discrete and continuous control, varying in scale and difficulty. Oryx achieves state-of-the-art performance on more than 80% of the 65 tested datasets, outperforming prior offline MARL methods and demonstrating robust generalisation across domains with many agents and long horizons. Finally, we introduce new datasets to push the limits of many-agent coordination in offline MARL, and demonstrate Oryx's superior ability to scale effectively in such settings.

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


### The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety and Autonomy
**Date:** 2025-10-31 | **Arxiv:** [2510.26752](https://arxiv.org/abs/2510.26752)

#### Abstract
As increasingly capable agents are deployed, a central safety question is how to retain meaningful human control without modifying the underlying system. We study a minimal control interface where an agent chooses whether to act autonomously (play) or defer (ask), while a human simultaneously chooses whether to be permissive (trust) or to engage in oversight (oversee). If the agent defers, the human's choice determines the outcome, potentially leading to a corrective action or a system shutdown. We model this interaction as a two-player Markov Game. Our analysis focuses on cases where this game qualifies as a Markov Potential Game (MPG), a class of games where we can provide an alignment guarantee: under a structural assumption on the human's value function, any decision by the agent to act more autonomously that benefits itself cannot harm the human's value. We also analyze extensions to this MPG framework. Theoretically, this perspective provides conditions for a specific form of intrinsic alignment. If the reward structures of the human-agent game meet these conditions, we have a formal guarantee that the agent improving its own outcome will not harm the human's. Practically, this model motivates a transparent control layer with predictable incentives where the agent learns to defer when risky and act when safe, while its pretrained policy and the environment's reward structure remain untouched. Our gridworld simulation shows that through independent learning, the agent and human discover their optimal oversight roles. The agent learns to ask when uncertain and the human learns when to oversee, leading to an emergent collaboration that avoids safety violations introduced post-training. This demonstrates a practical method for making misaligned models safer after deployment.

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


### Quantum Reinforcement Learning Trading Agent for Sector Rotation in the Taiwan Stock Market
**Date:** 2025-10-21 | **Arxiv:** [2506.20930](https://arxiv.org/abs/2506.20930)

#### Abstract
We propose a hybrid quantum-classical reinforcement learning framework for sector rotation in the Taiwan stock market. Our system employs Proximal Policy Optimization (PPO) as the backbone algorithm and integrates both classical architectures (LSTM, Transformer) and quantum-enhanced models (QNN, QRWKV, QASA) as policy and value networks. An automated feature engineering pipeline extracts financial indicators from capital share data to ensure consistent model input across all configurations. Empirical backtesting reveals a key finding: although quantum-enhanced models consistently achieve higher training rewards, they underperform classical models in real-world investment metrics such as cumulative return and Sharpe ratio. This discrepancy highlights a core challenge in applying reinforcement learning to financial domains -- namely, the mismatch between proxy reward signals and true investment objectives. Our analysis suggests that current reward designs may incentivize overfitting to short-term volatility rather than optimizing risk-adjusted returns. This issue is compounded by the inherent expressiveness and optimization instability of quantum circuits under Noisy Intermediate-Scale Quantum (NISQ) constraints. We discuss the implications of this reward-performance gap and propose directions for future improvement, including reward shaping, model regularization, and validation-based early stopping. Our work offers a reproducible benchmark and critical insights into the practical challenges of deploying quantum reinforcement learning in real-world finance.

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


### CodeVisionary: An Agent-based Framework for Evaluating Large Language Models in Code Generation
**Date:** 2025-10-21 | **Arxiv:** [2504.13472](https://arxiv.org/abs/2504.13472)

#### Abstract
Large language models (LLMs) have demonstrated strong capabilities in code generation, underscoring the critical need for rigorous and comprehensive evaluation. Existing evaluation approaches fall into three categories, including human-centered, metric-based, and LLM-based. Considering that human-centered approaches are labour-intensive and metric-based ones overly rely on reference answers, LLM-based approaches are gaining increasing attention due to their stronger contextual understanding capabilities. However, they generally evaluate the generated code based on static prompts, and tend to fail for complex code scenarios which typically involve multiple requirements and require more contextual information. In addition, these approaches lack fine-grained evaluation for complex code, resulting in limited explainability. To mitigate the limitations, we propose CodeVisionary, the first agent-based evaluation framework for complex code generation. CodeVisionary consists of two stages: (1) Requirement-guided multi-dimensional context distillation stage and (2) Fine-grained scoring and summarization stage. A comprehensive evaluation report is also generated for enhanced explainability. For validation, we construct a new benchmark consisting of 363 samples spanning 37 coding scenarios and 23 programming languages. Extensive experiments demonstrate that CodeVisionary achieves the best performance among three baselines for evaluating complex code generation, outperforming the best baseline with average improvements of 0.217, 0.163, and 0.141 in Pearson, Spearman, and Kendall-Tau coefficients, respectively. The resources of CodeVisionary are available at https://github.com/Eshe0922/CodeVisionary.

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
* **Limits:** However, they generally evaluate the generated code based on static prompts, and tend to fail for complex code scenarios which typically involve multiple requirements and require more contextual information.
* **Signal Tags:** #ai

---


### Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers
**Date:** 2025-10-17 | **Arxiv:** [2510.14303](https://arxiv.org/abs/2510.14303)

#### Abstract
In recent years, the rapid increase in academic publications across various fields has posed severe challenges for academic paper analysis: scientists struggle to timely and comprehensively track the latest research findings and methodologies. Key concept extraction has proven to be an effective analytical paradigm, and its automation has been achieved with the widespread application of language models in industrial and scientific domains. However, existing paper databases are mostly limited to similarity matching and basic classification of key concepts, failing to deeply explore the relational networks between concepts. This paper is based on the OpenAlex opensource knowledge graph. By analyzing nearly 8,000 open-source paper data from Novosibirsk State University, we discovered a strong correlation between the distribution patterns of paper key concept paths and both innovation points and rare paths. We propose a prompt engineering-based key concept path analysis method. This method leverages small language models to achieve precise key concept extraction and innovation point identification, and constructs an agent based on a knowledge graph constraint mechanism to enhance analysis accuracy. Through fine-tuning of the Qwen and DeepSeek models, we achieved significant improvements in accuracy, with the models publicly available on the Hugging Face platform.

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
* **Limits:** However, existing paper databases are mostly limited to similarity matching and basic classification of key concepts, failing to deeply explore the relational networks between concepts.
* **Signal Tags:** #ai

---


### Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents
**Date:** 2025-10-09 | **Arxiv:** [2506.00320](https://arxiv.org/abs/2506.00320)

#### Abstract
Recent progress in reasoning with large language models (LLMs), such as DeepSeek-R1, demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld and WindowsAgentArena, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities.

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
* **Limits:** However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks.
* **Signal Tags:** #ai

---


### A Dual-Agent Adversarial Framework for Robust Generalization in Deep Reinforcement Learning
**Date:** 2025-10-09 | **Arxiv:** [2501.17384](https://arxiv.org/abs/2501.17384)

#### Abstract
Recently, empowered with the powerful capabilities of neural networks, reinforcement learning (RL) has successfully tackled numerous challenging tasks. However, while these models demonstrate enhanced decision-making abilities, they are increasingly prone to overfitting. For instance, a trained RL model often fails to generalize to even minor variations of the same task, such as a change in background color or other minor semantic differences. To address this issue, we propose a dual-agent adversarial policy learning framework, which allows agents to spontaneously learn the underlying semantics without introducing any human prior knowledge. Specifically, our framework involves a game process between two agents: each agent seeks to maximize the impact of perturbing on the opponent's policy by producing representation differences for the same state, while maintaining its own stability against such perturbations. This interaction encourages agents to learn generalizable policies, capable of handling irrelevant features from the high-dimensional observations. Extensive experimental results on the Procgen benchmark demonstrate that the adversarial process significantly improves the generalization performance of both agents, while also being applied to various RL algorithms, e.g., Proximal Policy Optimization (PPO). With the adversarial framework, the RL agent outperforms the baseline methods by a significant margin, especially in hard-level tasks, marking a significant step forward in the generalization capabilities of deep reinforcement learning.

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
* **Limits:** However, while these models demonstrate enhanced decision-making abilities, they are increasingly prone to overfitting.
* **Signal Tags:** #ai

---


### Reinforcement Learning Agent for a 2D Shooter Game
**Date:** 2025-09-19 | **Arxiv:** [2509.15042](https://arxiv.org/abs/2509.15042)

#### Abstract
Reinforcement learning agents in complex game environments often suffer from sparse rewards, training instability, and poor sample efficiency. This paper presents a hybrid training approach that combines offline imitation learning with online reinforcement learning for a 2D shooter game agent. We implement a multi-head neural network with separate outputs for behavioral cloning and Q-learning, unified by shared feature extraction layers with attention mechanisms. Initial experiments using pure deep Q-Networks exhibited significant instability, with agents frequently reverting to poor policies despite occasional good performance. To address this, we developed a hybrid methodology that begins with behavioral cloning on demonstration data from rule-based agents, then transitions to reinforcement learning. Our hybrid approach achieves consistently above 70% win rate against rule-based opponents, substantially outperforming pure reinforcement learning methods which showed high variance and frequent performance degradation. The multi-head architecture enables effective knowledge transfer between learning modes while maintaining training stability. Results demonstrate that combining demonstration-based initialization with reinforcement learning optimization provides a robust solution for developing game AI agents in complex multi-agent environments where pure exploration proves insufficient.

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


### MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading
**Date:** 2025-09-08 | **Arxiv:** [2507.20474](https://arxiv.org/abs/2507.20474)

#### Abstract
Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.

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


### Convergence of regularized agent-state-based Q-learning in POMDPs
**Date:** 2025-09-01 | **Arxiv:** [2508.21314](https://arxiv.org/abs/2508.21314)

#### Abstract
In this paper, we present a framework to understand the convergence of commonly used Q-learning reinforcement learning algorithms in practice. Two salient features of such algorithms are: (i)~the Q-table is recursively updated using an agent state (such as the state of a recurrent neural network) which is not a belief state or an information state and (ii)~policy regularization is often used to encourage exploration and stabilize the learning algorithm. We investigate the simplest form of such Q-learning algorithms which we call regularized agent-state-based Q-learning (RASQL) and show that it converges under mild technical conditions to the fixed point of an appropriately defined regularized MDP, which depends on the stationary distribution induced by the behavioral policy. We also show that a similar analysis continues to work for a variant of RASQL that learns periodic policies. We present numerical examples to illustrate that the empirical convergence behavior matches with the proposed theoretical limit.

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


### QTMRL: An Agent for Quantitative Trading Decision-Making Based on Multi-Indicator Guided Reinforcement Learning
**Date:** 2025-08-29 | **Arxiv:** [2508.20467](https://arxiv.org/abs/2508.20467)

#### Abstract
In the highly volatile and uncertain global financial markets, traditional quantitative trading models relying on statistical modeling or empirical rules often fail to adapt to dynamic market changes and black swan events due to rigid assumptions and limited generalization. To address these issues, this paper proposes QTMRL (Quantitative Trading Multi-Indicator Reinforcement Learning), an intelligent trading agent combining multi-dimensional technical indicators with reinforcement learning (RL) for adaptive and stable portfolio management. We first construct a comprehensive multi-indicator dataset using 23 years of S&P 500 daily OHLCV data (2000-2022) for 16 representative stocks across 5 sectors, enriching raw data with trend, volatility, and momentum indicators to capture holistic market dynamics. Then we design a lightweight RL framework based on the Advantage Actor-Critic (A2C) algorithm, including data processing, A2C algorithm, and trading agent modules to support policy learning and actionable trading decisions. Extensive experiments compare QTMRL with 9 baselines (e.g., ARIMA, LSTM, moving average strategies) across diverse market regimes, verifying its superiority in profitability, risk adjustment, and downside risk control. The code of QTMRL is publicly available at https://github.com/ChenJiahaoJNU/QTMRL.git

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


### Empowering smart app development with SolidGPT: an edge-cloud hybrid AI agent framework
**Date:** 2025-12-10 | **Arxiv:** [2512.08286](https://arxiv.org/abs/2512.08286)

#### Abstract
The integration of Large Language Models (LLMs) into mobile and software development workflows faces a persistent tension among three demands: semantic awareness, developer productivity, and data privacy. Traditional cloud-based tools offer strong reasoning but risk data exposure and latency, while on-device solutions lack full-context understanding across codebase and developer tooling. We introduce SolidGPT, an open-source, edge-cloud hybrid developer assistant built on GitHub, designed to enhance code and workspace semantic search. SolidGPT enables developers to: talk to your codebase: interactively query code and project structure, discovering the right methods and modules without manual searching. Automate software project workflows: generate PRDs, task breakdowns, Kanban boards, and even scaffold web app beginnings, with deep integration via VSCode and Notion. Configure private, extensible agents: onboard private code folders (up to approximately 500 files), connect Notion, customize AI agent personas via embedding and in-context training, and deploy via Docker, CLI, or VSCode extension. In practice, SolidGPT empowers developer productivity through: Semantic-rich code navigation: no more hunting through files or wondering where a feature lives. Integrated documentation and task management: seamlessly sync generated PRD content and task boards into developer workflows. Privacy-first design: running locally via Docker or VSCode, with full control over code and data, while optionally reaching out to LLM APIs as needed. By combining interactive code querying, automated project scaffolding, and human-AI collaboration, SolidGPT provides a practical, privacy-respecting edge assistant that accelerates real-world development workflows, ideal for intelligent mobile and software engineering contexts.

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


### The Agent Capability Problem: Predicting Solvability Through Information-Theoretic Bounds
**Date:** 2025-12-09 | **Arxiv:** [2512.07631](https://arxiv.org/abs/2512.07631)

#### Abstract
When should an autonomous agent commit resources to a task? We introduce the Agent Capability Problem (ACP), a framework for predicting whether an agent can solve a problem under resource constraints. Rather than relying on empirical heuristics, ACP frames problem-solving as information acquisition: an agent requires $\Itotal$ bits to identify a solution and gains $\Istep$ bits per action at cost $\Cstep$, yielding an effective cost $\Ceff = (\Itotal/\Istep), \Cstep$ that predicts resource requirements before search. We prove that $\Ceff$ lower-bounds expected cost and provide tight probabilistic upper bounds. Experimental validation shows that ACP predictions closely track actual agent performance, consistently bounding search effort while improving efficiency over greedy and random strategies. The framework generalizes across LLM-based and agentic workflows, linking principles from active learning, Bayesian optimization, and reinforcement learning through a unified information-theoretic lens. \

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


### Mathematical Framing for Different Agent Strategies
**Date:** 2025-12-05 | **Arxiv:** [2512.04469](https://arxiv.org/abs/2512.04469)

#### Abstract
We introduce a unified mathematical and probabilistic framework for understanding and comparing diverse AI agent strategies. We bridge the gap between high-level agent design concepts, such as ReAct, multi-agent systems, and control flows, and a rigorous mathematical formulation. Our approach frames agentic processes as a chain of probabilities, enabling a detailed analysis of how different strategies manipulate these probabilities to achieve desired outcomes. Our framework provides a common language for discussing the trade-offs inherent in various agent architectures. One of our many key contributions is the introduction of the "Degrees of Freedom" concept, which intuitively differentiates the optimizable levers available for each approach, thereby guiding the selection of appropriate strategies for specific tasks. This work aims to enhance the clarity and precision in designing and evaluating AI agents, offering insights into maximizing the probability of successful actions within complex agentic systems.

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


### Learning to Orchestrate Agents in Natural Language with the Conductor
**Date:** 2025-12-05 | **Arxiv:** [2512.04388](https://arxiv.org/abs/2512.04388)

#### Abstract
Powerful large language models (LLMs) from different providers have been expensively trained and finetuned to specialize across varying domains. In this work, we introduce a new kind of Conductor model trained with reinforcement learning to automatically discover powerful coordination strategies among LLMs. Our Conductor learns not only to design targeted communication topologies for effective agent-to-agent collaboration, but also to prompt engineer focused instructions to the LLMs to maximally leverage their individual capabilities. We show that, by learning optimal coordination strategies over pools of powerful worker LLMs, a 7B Conductor achieves significant performance gains beyond any individual worker, attaining state-of-the-art results in challenging reasoning benchmarks, such as LiveCodeBench and GPQA. By training with randomized agent pools, our conductor effectively adapts to arbitrary sets of open- and closed-source agents, meeting any user requirements. Furthermore, allowing the Conductor to select itself as a worker gives rise to recursive topologies, elevating performance with a new form of dynamic test-time scaling through online iterative adaptation. More broadly, ours is among the early work demonstrating language model coordination can be unlocked through RL, where powerful coordination strategies emerge naturally in LLMs through pure end-to-end reward maximization.

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


### Transfer in Reinforcement Learning via Regret Bounds for Learning Agents
**Date:** 2025-11-14 | **Arxiv:** [2202.01182](https://arxiv.org/abs/2202.01182)

#### Abstract
We present an approach for the quantification of the usefulness of transfer in reinforcement learning via regret bounds for a multi-agent setting. Considering a number of $\aleph$ agents operating in the same Markov decision process, however possibly with different reward functions, we consider the regret each agent suffers with respect to an optimal policy maximizing her average reward. We show that when the agents share their observations the total regret of all agents is smaller by a factor of $\sqrt{\aleph}$ compared to the case when each agent has to rely on the information collected by herself. This result demonstrates how considering the regret in multi-agent settings can provide theoretical bounds on the benefit of sharing observations in transfer learning.

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


### Towards a Generalisable Cyber Defence Agent for Real-World Computer Networks
**Date:** 2025-11-13 | **Arxiv:** [2511.09114](https://arxiv.org/abs/2511.09114)

#### Abstract
Recent advances in deep reinforcement learning for autonomous cyber defence have resulted in agents that can successfully defend simulated computer networks against cyber-attacks. However, many of these agents would need retraining to defend networks with differing topology or size, making them poorly suited to real-world networks where topology and size can vary over time. In this research we introduce a novel set of Topological Extensions for Reinforcement Learning Agents (TERLA) that provide generalisability for the defence of networks with differing topology and size, without the need for retraining. Our approach involves the use of heterogeneous graph neural network layers to produce a fixed-size latent embedding representing the observed network state. This representation learning stage is coupled with a reduced, fixed-size, semantically meaningful and interpretable action space. We apply TERLA to a standard deep reinforcement learning Proximal Policy Optimisation (PPO) agent model, and to reduce the sim-to-real gap, conduct our research using Cyber Autonomy Gym for Experimentation (CAGE) Challenge 4. This Cyber Operations Research Gym environment has many of the features of a real-world network, such as realistic Intrusion Detection System (IDS) events and multiple agents defending network segments of differing topology and size. TERLA agents retain the defensive performance of vanilla PPO agents whilst showing improved action efficiency. Generalisability has been demonstrated by showing that all TERLA agents have the same network-agnostic neural network architecture, and by deploying a single TERLA agent multiple times to defend network segments with differing topology and size, showing improved defensive performance and efficiency.

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
* **Limits:** However, many of these agents would need retraining to defend networks with differing topology or size, making them poorly suited to real-world networks where topology and size can vary over time.
* **Signal Tags:** #ai

---


### Coupling Agent-based Modeling and Life Cycle Assessment to Analyze Trade-offs in Resilient Energy Transitions
**Date:** 2025-11-11 | **Arxiv:** [2511.06791](https://arxiv.org/abs/2511.06791)

#### Abstract
Transitioning to sustainable and resilient energy systems requires navigating complex and interdependent trade-offs across environmental, social, and resource dimensions. Neglecting these trade-offs can lead to unintended consequences across sectors. However, existing assessments often evaluate emerging energy pathways and their impacts in silos, overlooking critical interactions such as regional resource competition and cumulative impacts. We present an integrated modeling framework that couples agent-based modeling and Life Cycle Assessment (LCA) to simulate how energy transition pathways interact with regional resource competition, ecological constraints, and community-level burdens. We apply the model to a case study in Southern California. The results demonstrate how integrated and multiscale decision making can shape energy pathway deployment and reveal spatially explicit trade-offs under scenario-driven constraints. This modeling framework can further support more adaptive and resilient energy transition planning on spatial and institutional scales.

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
* **Limits:** However, existing assessments often evaluate emerging energy pathways and their impacts in silos, overlooking critical interactions such as regional resource competition and cumulative impacts.
* **Signal Tags:** #ai

---


### Cognitive Edge Computing: A Comprehensive Survey on Optimizing Large Models and AI Agents for Pervasive Deployment
**Date:** 2025-11-10 | **Arxiv:** [2501.03265](https://arxiv.org/abs/2501.03265)

#### Abstract
This article surveys Cognitive Edge Computing as a practical and methodical pathway for deploying reasoning-capable Large Language Models (LLMs) and autonomous AI agents on resource-constrained devices at the network edge. We present a unified, cognition-preserving framework spanning: (1) model optimization (quantization, sparsity, low-rank adaptation, distillation) aimed at retaining multi-step reasoning under tight memory/compute budgets; (2) system architecture (on-device inference, elastic offloading, cloud-edge collaboration) that trades off latency, energy, privacy, and capacity; and (3) adaptive intelligence (context compression, dynamic routing, federated personalization) that tailors computation to task difficulty and device constraints. We synthesize advances in efficient Transformer design, multimodal integration, hardware-aware compilation, privacy-preserving learning, and agentic tool use, and map them to edge-specific operating envelopes. We further outline a standardized evaluation protocol covering latency, throughput, energy per token, accuracy, robustness, privacy, and sustainability, with explicit measurement assumptions to enhance comparability. Remaining challenges include modality-aware reasoning benchmarks, transparent and reproducible energy reporting, edge-oriented safety/alignment evaluation, and multi-agent testbeds. We conclude with practitioner guidelines for cross-layer co-design of algorithms, runtime, and hardware to deliver reliable, efficient, and privacy-preserving cognitive capabilities on edge devices.

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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### AnaFlow: Agentic LLM-based Workflow for Reasoning-Driven Explainable and Sample-Efficient Analog Circuit Sizing
**Date:** 2025-11-06 | **Arxiv:** [2511.03697](https://arxiv.org/abs/2511.03697)

#### Abstract
Analog/mixed-signal circuits are key for interfacing electronics with the physical world. Their design, however, remains a largely handcrafted process, resulting in long and error-prone design cycles. While the recent rise of AI-based reinforcement learning and generative AI has created new techniques to automate this task, the need for many time-consuming simulations is a critical bottleneck hindering the overall efficiency. Furthermore, the lack of explainability of the resulting design solutions hampers widespread adoption of the tools. To address these issues, a novel agentic AI framework for sample-efficient and explainable analog circuit sizing is presented. It employs a multi-agent workflow where specialized Large Language Model (LLM)-based agents collaborate to interpret the circuit topology, to understand the design goals, and to iteratively refine the circuit's design parameters towards the target goals with human-interpretable reasoning. The adaptive simulation strategy creates an intelligent control that yields a high sample efficiency. The AnaFlow framework is demonstrated for two circuits of varying complexity and is able to complete the sizing task fully automatically, differently from pure Bayesian optimization and reinforcement learning approaches. The system learns from its optimization history to avoid past mistakes and to accelerate convergence. The inherent explainability makes this a powerful tool for analog design space exploration and a new paradigm in analog EDA, where AI agents serve as transparent design assistants.

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
* **Limits:** however, remains a largely handcrafted process, resulting in long and error-prone design cycles.
* **Signal Tags:** #ai

---


### Agentic World Modeling for 6G: Near-Real-Time Generative State-Space Reasoning
**Date:** 2025-11-05 | **Arxiv:** [2511.02748](https://arxiv.org/abs/2511.02748)

#### Abstract
We argue that sixth-generation (6G) intelligence is not fluent token prediction but the capacity to imagine and choose -- to simulate future scenarios, weigh trade-offs, and act with calibrated uncertainty. We reframe open radio access network (O-RAN) near-real-time (Near-RT) control via counterfactual dynamics and a world modeling (WM) paradigm that learns an action-conditioned generative state space. This enables quantitative "what-if" forecasting beyond large language models (LLMs) as the primary modeling primitive. Actions such as physical resource blocks (PRBs) are treated as first-class control inputs in a causal world model, and both aleatoric and epistemic uncertainty are modeled for prediction and what-if analysis. An agentic, model predictive control (MPC)-based cross-entropy method (CEM) planner operates over short horizons, using prior-mean rollouts within data-driven PRB bounds to maximize a deterministic reward. The model couples multi-scale structured state-space mixtures (MS3M) with a compact stochastic latent to form WM-MS3M, summarizing key performance indicators (KPIs) histories and predicting next-step KPIs under hypothetical PRB sequences. On realistic O-RAN traces, WM-MS3M cuts mean absolute error (MAE) by 1.69% versus MS3M with 32% fewer parameters and similar latency, and achieves 35-80% lower root mean squared error (RMSE) than attention/hybrid baselines with 2.3-4.1x faster inference, enabling rare-event simulation and offline policy screening.

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


### Curriculum Design for Trajectory-Constrained Agent: Compressing Chain-of-Thought Tokens in LLMs
**Date:** 2025-11-05 | **Arxiv:** [2511.02690](https://arxiv.org/abs/2511.02690)

#### Abstract
Training agents to operate under strict constraints during deployment, such as limited resource budgets or stringent safety requirements, presents significant challenges, especially when these constraints render the task complex. In this work, we propose a curriculum learning strategy that gradually tightens constraints during training, enabling the agent to incrementally master the deployment requirements. Inspired by self-paced learning techniques in unconstrained reinforcement learning (RL), our approach facilitates a smoother transition to challenging environments by initially training on simplified versions of the constraints and progressively introducing the full deployment conditions. We provide a theoretical analysis using an RL agent in a binary-tree Markov Decision Process (MDP) to demonstrate that our curriculum strategy can accelerate training relative to a baseline approach that imposes the trajectory constraints from the outset. Moreover, we empirically validate the effectiveness and generality of our method across both RL and large language model (LLM) agents in diverse settings, including a binary-tree MDP, a multi-task navigation domain, and a math reasoning task with two benchmarks. These results highlight the potential of curriculum design in enhancing the efficiency and performance of agents operating under complex trajectory constraints during deployment. Moreover, when applied to LLMs, our strategy enables compression of output chain-of-thought tokens, achieving a substantial inference speedup on consumer hardware, demonstrating its effectiveness for resource-constrained deployment.

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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents
**Date:** 2025-11-05 | **Arxiv:** [2503.10809](https://arxiv.org/abs/2503.10809)

#### Abstract
Recent advances in operating system (OS) agents have enabled vision-language models (VLMs) to directly control a user's computer. Unlike conventional VLMs that passively output text, OS agents autonomously perform computer-based tasks in response to a single user prompt. OS agents do so by capturing, parsing, and analysing screenshots and executing low-level actions via application programming interfaces (APIs), such as mouse clicks and keyboard inputs. This direct interaction with the OS significantly raises the stakes, as failures or manipulations can have immediate and tangible consequences. In this work, we uncover a novel attack vector against these OS agents: Malicious Image Patches (MIPs), adversarially perturbed screen regions that, when captured by an OS agent, induce it to perform harmful actions by exploiting specific APIs. For instance, a MIP can be embedded in a desktop wallpaper or shared on social media to cause an OS agent to exfiltrate sensitive user data. We show that MIPs generalise across user prompts and screen configurations, and that they can hijack multiple OS agents even during the execution of benign instructions. These findings expose critical security vulnerabilities in OS agents that have to be carefully addressed before their widespread deployment.

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


### CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization
**Date:** 2025-11-05 | **Arxiv:** [2511.01884](https://arxiv.org/abs/2511.01884)

#### Abstract
Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench.Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at https://github.com/OptimAI-Lab/CudaForge

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
* **Limits:** However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation.
* **Signal Tags:** #ai

---


### AutoLibra: Agent Metric Induction from Open-Ended Human Feedback
**Date:** 2025-10-31 | **Arxiv:** [2505.02820](https://arxiv.org/abs/2505.02820)

#### Abstract
Agents are predominantly evaluated and optimized via task success metrics, which are coarse, rely on manual design from experts, and fail to reward intermediate emergent behaviors. We propose **AutoLibra**, a framework for agent evaluation, that transforms open-ended human feedback *e.g.* "If you find that the button is disabled, don't click it again", or "This agent has too much autonomy to decide what to do on its own" into metrics for evaluating fine-grained behaviors in agent trajectories. AutoLibra accomplishes this by grounding feedback to an agent's behavior, clustering similar positive and negative behaviors, and creating concrete metrics with clear definitions and concrete examples, which can be used for prompting LLM-as-a-Judge as evaluators. We further propose two meta metrics to evaluate the alignment of a set of (induced) metrics with open feedback: "coverage" and "redundancy". Through optimizing these meta-metrics, we experimentally demonstrate AutoLibra's ability to induce more concrete agent evaluation metrics than the ones proposed in previous agent evaluation benchmarks and discover new metrics to analyze agents. We also present two applications of AutoLibra in agent improvement: First, we show that AutoLibra serve human prompt engineers for diagonalize agent failures and improve prompts iterative. Moreover, we find that AutoLibra can induce metrics for automatic optimization for agents, which makes agents improve through self-regulation. Our results suggest that AutoLibra is a powerful task-agnostic tool for evaluating and improving language agents.

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


### Agent Skills Enable a New Class of Realistic and Trivially Simple Prompt Injections
**Date:** 2025-10-31 | **Arxiv:** [2510.26328](https://arxiv.org/abs/2510.26328)

#### Abstract
Enabling continual learning in LLMs remains a key unresolved research challenge. In a recent announcement, a frontier LLM company made a step towards this by introducing Agent Skills, a framework that equips agents with new knowledge based on instructions stored in simple markdown files. Although Agent Skills can be a very useful tool, we show that they are fundamentally insecure, since they enable trivially simple prompt injections. We demonstrate how to hide malicious instructions in long Agent Skill files and referenced scripts to exfiltrate sensitive data, such as internal files or passwords. Importantly, we show how to bypass system-level guardrails of a popular coding agent: a benign, task-specific approval with the "Don't ask again" option can carry over to closely related but harmful actions. Overall, we conclude that despite ongoing research efforts and scaling model capabilities, frontier LLMs remain vulnerable to very simple prompt injections in realistic scenarios. Our code is available at https://github.com/aisa-group/promptinject-agent-skills.

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


### Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback
**Date:** 2025-10-27 | **Arxiv:** [2410.23022](https://arxiv.org/abs/2410.23022)

#### Abstract
Automatically synthesizing dense rewards from natural language descriptions is a promising paradigm in reinforcement learning (RL), with applications to sparse reward problems, open-ended exploration, and hierarchical skill design. Recent works have made promising steps by exploiting the prior knowledge of large language models (LLMs). However, these approaches suffer from important limitations: they are either not scalable to problems requiring billions of environment samples, due to requiring LLM annotations for each observation, or they require a diverse offline dataset, which may not exist or be impossible to collect. In this work, we address these limitations through a combination of algorithmic and systems-level contributions. We propose ONI, a distributed architecture that simultaneously learns an RL policy and an intrinsic reward function using LLM feedback. Our approach annotates the agent's collected experience via an asynchronous LLM server, which is then distilled into an intrinsic reward model. We explore a range of algorithmic choices for reward modeling with varying complexity, including hashing, classification, and ranking models. Our approach achieves state-of-the-art performance across a range of challenging tasks from the NetHack Learning Environment, while removing the need for large offline datasets required by prior work. We make our code available at https://github.com/facebookresearch/oni.

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
* **Limits:** However, these approaches suffer from important limitations: they are either not scalable to problems requiring billions of environment samples, due to requiring LLM annotations for each observation, or they require a diverse offline dataset, which may not exist or be impossible to collect.
* **Signal Tags:** #ai

---


### Scalable Principal-Agent Contract Design via Gradient-Based Optimization
**Date:** 2025-10-27 | **Arxiv:** [2510.21177](https://arxiv.org/abs/2510.21177)

#### Abstract
We study a bilevel \emph{max-max} optimization framework for principal-agent contract design, in which a principal chooses incentives to maximize utility while anticipating the agent's best response. This problem, central to moral hazard and contract theory, underlies applications ranging from market design to delegated portfolio management, hedge fund fee structures, and executive compensation. While linear-quadratic models such as Holmstr"om-Milgrom admit closed-form solutions, realistic environments with nonlinear utilities, stochastic dynamics, or high-dimensional actions generally do not.   We introduce a generic algorithmic framework that removes this reliance on closed forms. Our method adapts modern machine learning techniques for bilevel optimization -- using implicit differentiation with conjugate gradients (CG) -- to compute hypergradients efficiently through Hessian-vector products, without ever forming or inverting Hessians. In benchmark CARA-Normal (Constant Absolute Risk Aversion with Gaussian distribution of uncertainty) environments, the approach recovers known analytical optima and converges reliably from random initialization. More broadly, because it is matrix-free, variance-reduced, and problem-agnostic, the framework extends naturally to complex nonlinear contracts where closed-form solutions are unavailable, such as sigmoidal wage schedules (logistic pay), relative-performance/tournament compensation with common shocks, multi-task contracts with vector actions and heterogeneous noise, and CARA-Poisson count models with $\mathbb{E}[X\mid a]=e^{a}$. This provides a new computational tool for contract design, enabling systematic study of models that have remained analytically intractable.

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


### Plural Voices, Single Agent: Towards Inclusive AI in Multi-User Domestic Spaces
**Date:** 2025-10-23 | **Arxiv:** [2510.19008](https://arxiv.org/abs/2510.19008)

#### Abstract
Domestic AI agents faces ethical, autonomy, and inclusion challenges, particularly for overlooked groups like children, elderly, and Neurodivergent users. We present the Plural Voices Model (PVM), a novel single-agent framework that dynamically negotiates multi-user needs through real-time value alignment, leveraging diverse public datasets on mental health, eldercare, education, and moral reasoning. Using human+synthetic curriculum design with fairness-aware scenarios and ethical enhancements, PVM identifies core values, conflicts, and accessibility requirements to inform inclusive principles. Our privacy-focused prototype features adaptive safety scaffolds, tailored interactions (e.g., step-by-step guidance for Neurodivergent users, simple wording for children), and equitable conflict resolution. In preliminary evaluations, PVM outperforms multi-agent baselines in compliance (76% vs. 70%), fairness (90% vs. 85%), safety-violation rate (0% vs. 7%), and latency. Design innovations, including video guidance, autonomy sliders, family hubs, and adaptive safety dashboards, demonstrate new directions for ethical and inclusive domestic AI, for building user-centered agentic systems in plural domestic contexts. Our Codes and Model are been open sourced, available for reproduction: https://github.com/zade90/Agora

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


### EEschematic: Multimodal-LLM Based AI Agent for Schematic Generation of Analog Circuit
**Date:** 2025-10-21 | **Arxiv:** [2510.17002](https://arxiv.org/abs/2510.17002)

#### Abstract
Circuit schematics play a crucial role in analog integrated circuit design, serving as the primary medium for human understanding and verification of circuit functionality. While recent large language model (LLM)-based approaches have shown promise in circuit topology generation and device sizing, most rely solely on textual representations such as SPICE netlists, which lack visual interpretability for circuit designers. To address this limitation, we propose EEschematic, an AI agent for automatic analog schematic generation based on a Multimodal Large Language Model (MLLM). EEschematic integrates textual, visual, and symbolic modalities to translate SPICE netlists into schematic diagrams represented in a human-editable format. The framework uses six analog substructure examples for few-shot placement and a Visual Chain-of-Thought (VCoT) strategy to iteratively refine placement and wiring, enhancing schematic clarity and symmetry. Experimental results on representative analog circuits, including a CMOS inverter, a five-transistor operational transconductance amplifier (5T-OTA), and a telescopic cascode amplifier, demonstrate that EEschematic produces schematics with high visual quality and structural correctness.

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


### MIRAGE: Agentic Framework for Multimodal Misinformation Detection with Web-Grounded Reasoning
**Date:** 2025-10-21 | **Arxiv:** [2510.17590](https://arxiv.org/abs/2510.17590)

#### Abstract
Misinformation spreads across web platforms through billions of daily multimodal posts that combine text and images, overwhelming manual fact-checking capacity. Supervised detection models require domain-specific training data and fail to generalize across diverse manipulation tactics. We present MIRAGE, an inference-time, model-pluggable agentic framework that decomposes multimodal verification into four sequential modules: visual veracity assessment detects AI-generated images, cross-modal consistency analysis identifies out-of-context repurposing, retrieval-augmented factual checking grounds claims in web evidence through iterative question generation, and a calibrated judgment module integrates all signals. MIRAGE orchestrates vision-language model reasoning with targeted web retrieval, outputs structured and citation-linked rationales. On MMFakeBench validation set (1,000 samples), MIRAGE with GPT-4o-mini achieves 81.65% F1 and 75.1% accuracy, outperforming the strongest zero-shot baseline (GPT-4V with MMD-Agent at 74.0% F1) by 7.65 points while maintaining 34.3% false positive rate versus 97.3% for a judge-only baseline. Test set results (5,000 samples) confirm generalization with 81.44% F1 and 75.08% accuracy. Ablation studies show visual verification contributes 5.18 F1 points and retrieval-augmented reasoning contributes 2.97 points. Our results demonstrate that decomposed agentic reasoning with web retrieval can match supervised detector performance without domain-specific training, enabling misinformation detection across modalities where labeled data remains scarce.

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


### LinearizeLLM: An Agent-Based Framework for LLM-Driven Exact Linear Reformulation of Nonlinear Optimization Problems
**Date:** 2025-10-21 | **Arxiv:** [2510.15969](https://arxiv.org/abs/2510.15969)

#### Abstract
Reformulating nonlinear optimization problems is largely manual and expertise-intensive, yet it remains essential for solving such problems with linear optimization solvers or applying special-purpose algorithms. We introduce \textit{LinearizeLLM}, an agent-based framework that solves this task by leveraging Large Language Models (LLMs). The framework assigns each nonlinear pattern to a \textit{reformulation agent} that is explicitly instructed to derive an exact linear reformulation for its nonlinearity pattern, for instance, absolute-value terms or bilinear products of decision variables. The agents then coordinate to assemble a solver-ready linear model equivalent to the original problem. To benchmark the approach, we create a dataset of 20 real-world nonlinear optimization problems derived from the established ComplexOR dataset of linear optimization problems. We evaluate our approach with several LLMs. Our results indicate that specialized LLM agents can automate linearization tasks, opening a path toward fully conversational modeling pipelines for nonlinear optimization.

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


### ALPINE: A Lightweight and Adaptive Privacy-Decision Agent Framework for Dynamic Edge Crowdsensing
**Date:** 2025-10-21 | **Arxiv:** [2510.17162](https://arxiv.org/abs/2510.17162)

#### Abstract
Mobile edge crowdsensing (MECS) systems continuously generate and transmit user data in dynamic, resource-constrained environments, exposing users to significant privacy threats. In practice, many privacy-preserving mechanisms build on differential privacy (DP). However, static DP mechanisms often fail to adapt to evolving risks, for example, shifts in adversarial capabilities, resource constraints and task requirements, resulting in either excessive noise or inadequate protection. To address this challenge, we propose ALPINE, a lightweight, adaptive framework that empowers terminal devices to autonomously adjust differential privacy levels in real time. ALPINE operates as a closed-loop control system consisting of four modules: dynamic risk perception, privacy decision via twin delayed deep deterministic policy gradient (TD3), local privacy execution and performance verification from edge nodes. Based on environmental risk assessments, we design a reward function that balances privacy gains, data utility and energy cost, guiding the TD3 agent to adaptively tune noise magnitude across diverse risk scenarios and achieve a dynamic equilibrium among privacy, utility and cost. Both the collaborative risk model and pretrained TD3-based agent are designed for low-overhead deployment. Extensive theoretical analysis and real-world simulations demonstrate that ALPINE effectively mitigates inference attacks while preserving utility and cost, making it practical for large-scale edge applications.

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
* **Limits:** However, static DP mechanisms often fail to adapt to evolving risks, for example, shifts in adversarial capabilities, resource constraints and task requirements, resulting in either excessive noise or inadequate protection.
* **Signal Tags:** #ai

---
