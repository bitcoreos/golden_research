# Vol 34 Reinforcement Learning   Control
*Enriched by BITCOREOS | Phase 4 Batch 7*

---

### Context-Aware Model-Based Reinforcement Learning for Autonomous Racing
**Date:** 2025-10-15 | **Arxiv:** [2510.11501](https://hub.bitwiki.org/t/context-aware-model-based-reinforcement-learning-for-autonomous-racing/16850)

#### Abstract
Autonomous vehicles have shown promising potential to be a groundbreaking technology for improving the safety of road users. For these vehicles, as well as many other safety-critical robotic technologies, to be deployed in real-world applications, we require algorithms that can generalize well to unseen scenarios and data. Model-based reinforcement learning algorithms (MBRL) have demonstrated state-of-the-art performance and data efficiency across a diverse set of domains. However, these algorithms have also shown susceptibility to changes in the environment and its transition dynamics.   In this work, we explore the performance and generalization capabilities of MBRL algorithms for autonomous driving, specifically in the simulated autonomous racing environment, Roboracer (formerly F1Tenth). We frame the head-to-head racing task as a learning problem using contextual Markov decision processes and parameterize the driving behavior of the adversaries using the context of the episode, thereby also parameterizing the transition and reward dynamics. We benchmark the behavior of MBRL algorithms in this environment and propose a novel context-aware extension of the existing literature, cMask. We demonstrate that context-aware MBRL algorithms generalize better to out-of-distribution adversary behaviors relative to context-free approaches. We also demonstrate that cMask displays strong generalization capabilities, as well as further performance improvement relative to other context-aware MBRL approaches when racing against adversaries with in-distribution behaviors.

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
* **Limits:** However, these algorithms have also shown susceptibility to changes in the environment and its transition dynamics.
* **Signal Tags:** #ai

---


### Dynamics-Decoupled Trajectory Alignment for Sim-to-Real Transfer in Reinforcement Learning for Autonomous Driving
**Date:** 2025-11-11 | **Arxiv:** [2511.07155](https://hub.bitwiki.org/t/dynamics-decoupled-trajectory-alignment-for-sim-to-real-transfer-in-reinforcement-learning-for-autonomous-driving/22795)

#### Abstract
Reinforcement learning (RL) has shown promise in robotics, but deploying RL on real vehicles remains challenging due to the complexity of vehicle dynamics and the mismatch between simulation and reality. Factors such as tire characteristics, road surface conditions, aerodynamic disturbances, and vehicle load make it infeasible to model real-world dynamics accurately, which hinders direct transfer of RL agents trained in simulation. In this paper, we present a framework that decouples motion planning from vehicle control through a spatial and temporal alignment strategy between a virtual vehicle and the real system. An RL agent is first trained in simulation using a kinematic bicycle model to output continuous control actions. Its behavior is then distilled into a trajectory-predicting agent that generates finite-horizon ego-vehicle trajectories, enabling synchronization between virtual and real vehicles. At deployment, a Stanley controller governs lateral dynamics, while longitudinal alignment is maintained through adaptive update mechanisms that compensate for deviations between virtual and real trajectories. We validate our approach on a real vehicle and demonstrate that the proposed alignment strategy enables robust zero-shot transfer of RL-based motion planning from simulation to reality, successfully decoupling high-level trajectory generation from low-level vehicle control.

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


### Tactical Decision Making for Autonomous Trucks by Deep Reinforcement Learning with Total Cost of Operation Based Reward
**Date:** 2025-11-10 | **Arxiv:** [2403.06524](https://hub.bitwiki.org/t/tactical-decision-making-for-autonomous-trucks-by-deep-reinforcement-learning-with-total-cost-of-operation-based-reward/22381)

#### Abstract
We develop a deep reinforcement learning framework for tactical decision making in an autonomous truck, specifically for Adaptive Cruise Control (ACC) and lane change maneuvers in a highway scenario. Our results demonstrate that it is beneficial to separate high-level decision-making processes and low-level control actions between the reinforcement learning agent and the low-level controllers based on physical models. In the following, we study optimizing the performance with a realistic and multi-objective reward function based on Total Cost of Operation (TCOP) of the truck using different approaches; by adding weights to reward components, by normalizing the reward components and by using curriculum learning techniques.

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


### End-to-End Framework Integrating Generative AI and Deep Reinforcement Learning for Autonomous Ultrasound Scanning
**Date:** 2025-11-04 | **Arxiv:** [2511.00114](https://hub.bitwiki.org/t/end-to-end-framework-integrating-generative-ai-and-deep-reinforcement-learning-for-autonomous-ultrasound-scanning/21264)

#### Abstract
Cardiac ultrasound (US) is among the most widely used diagnostic tools in cardiology for assessing heart health, but its effectiveness is limited by operator dependence, time constraints, and human error. The shortage of trained professionals, especially in remote areas, further restricts access. These issues underscore the need for automated solutions that can ensure consistent, and accessible cardiac imaging regardless of operator skill or location. Recent progress in artificial intelligence (AI), especially in deep reinforcement learning (DRL), has gained attention for enabling autonomous decision-making. However, existing DRL-based approaches to cardiac US scanning lack reproducibility, rely on proprietary data, and use simplified models. Motivated by these gaps, we present the first end-to-end framework that integrates generative AI and DRL to enable autonomous and reproducible cardiac US scanning. The framework comprises two components: (i) a conditional generative simulator combining Generative Adversarial Networks (GANs) with Variational Autoencoders (VAEs), that models the cardiac US environment producing realistic action-conditioned images; and (ii) a DRL module that leverages this simulator to learn autonomous, accurate scanning policies. The proposed framework delivers AI-driven guidance through expert-validated models that classify image type and assess quality, supports conditional generation of realistic US images, and establishes a reproducible foundation extendable to other organs. To ensure reproducibility, a publicly available dataset of real cardiac US scans is released. The solution is validated through several experiments. The VAE-GAN is benchmarked against existing GAN variants, with performance assessed using qualitative and quantitative approaches, while the DRL-based scanning system is evaluated under varying configurations to demonstrate effectiveness.

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
* **Limits:** However, existing DRL-based approaches to cardiac US scanning lack reproducibility, rely on proprietary data, and use simplified models.
* **Signal Tags:** #ai

---


### Bootstrapping Reinforcement Learning with Sub-optimal Policies for Autonomous Driving
**Date:** 2025-09-08 | **Arxiv:** [2509.04712](https://hub.bitwiki.org/t/bootstrapping-reinforcement-learning-with-sub-optimal-policies-for-autonomous-driving/8120)

#### Abstract
Automated vehicle control using reinforcement learning (RL) has attracted significant attention due to its potential to learn driving policies through environment interaction. However, RL agents often face training challenges in sample efficiency and effective exploration, making it difficult to discover an optimal driving strategy. To address these issues, we propose guiding the RL driving agent with a demonstration policy that need not be a highly optimized or expert-level controller. Specifically, we integrate a rule-based lane change controller with the Soft Actor Critic (SAC) algorithm to enhance exploration and learning efficiency. Our approach demonstrates improved driving performance and can be extended to other driving scenarios that can similarly benefit from demonstration-based guidance.

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
* **Limits:** However, RL agents often face training challenges in sample efficiency and effective exploration, making it difficult to discover an optimal driving strategy.
* **Signal Tags:** #ai

---


### Prompt-Driven Domain Adaptation for End-to-End Autonomous Driving via In-Context RL
**Date:** 2025-11-18 | **Arxiv:** [2511.12755](https://hub.bitwiki.org/t/prompt-driven-domain-adaptation-for-end-to-end-autonomous-driving-via-in-context-rl/24330)

#### Abstract
Despite significant progress and advances in autonomous driving, many end-to-end systems still struggle with domain adaptation (DA), such as transferring a policy trained under clear weather to adverse weather conditions. Typical DA strategies in the literature include collecting additional data in the target domain or re-training the model, or both. Both these strategies quickly become impractical as we increase scale and complexity of driving. These limitations have encouraged investigation into few-shot and zero-shot prompt-driven DA at inference time involving LLMs and VLMs. These methods work by adding a few state-action trajectories during inference to the prompt (similar to in-context learning). However, there are two limitations of such an approach: $(i)$ prompt-driven DA methods are currently restricted to perception tasks such as detection and segmentation and $(ii)$ they require expert few-shot data. In this work, we present a new approach to inference-time few-shot prompt-driven DA for closed-loop autonomous driving in adverse weather condition using in-context reinforcement learning (ICRL). Similar to other prompt-driven DA methods, our approach does not require any updates to the model parameters nor does it require additional data collection in adversarial weather regime. Furthermore, our approach advances the state-of-the-art in prompt-driven DA by extending to closed driving using general trajectories observed during inference. Our experiments using the CARLA simulator show that ICRL results in safer, more efficient, and more comfortable driving policies in the target domain compared to state-of-the-art prompt-driven DA baselines.

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
* **Limits:** However, there are two limitations of such an approach: $(i)$ prompt-driven DA methods are currently restricted to perception tasks such as detection and segmentation and $(ii)$ they require expert few-shot data.
* **Signal Tags:** #ai

---
