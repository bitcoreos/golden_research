# Vol 14 General Scientific ML
*Enriched by BITCOREOS | Phase 4 Batch 3*

---

### On the Stability of Neural Networks in Deep Learning
**Date:** 2025-10-30 | **Arxiv:** [2510.25282](https://arxiv.org/abs/2510.25282)

#### Abstract
Deep learning has achieved remarkable success across a wide range of tasks, but its models often suffer from instability and vulnerability: small changes to the input may drastically affect predictions, while optimization can be hindered by sharp loss landscapes. This thesis addresses these issues through the unifying perspective of sensitivity analysis, which examines how neural networks respond to perturbations at both the input and parameter levels.   We study Lipschitz networks as a principled way to constrain sensitivity to input perturbations, thereby improving generalization, adversarial robustness, and training stability. To complement this architectural approach, we introduce regularization techniques based on the curvature of the loss function, promoting smoother optimization landscapes and reducing sensitivity to parameter variations. Randomized smoothing is also explored as a probabilistic method for enhancing robustness at decision boundaries.   By combining these perspectives, we develop a unified framework where Lipschitz continuity, randomized smoothing, and curvature regularization interact to address fundamental challenges in stability. The thesis contributes both theoretical analysis and practical methodologies, including efficient spectral norm computation, novel Lipschitz-constrained layers, and improved certification procedures.

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


### Idea2Plan: Exploring AI-Powered Research Planning
**Date:** 2025-10-30 | **Arxiv:** [2510.24891](https://arxiv.org/abs/2510.24891)

#### Abstract
Large language models (LLMs) have demonstrated significant potential to accelerate scientific discovery as valuable tools for analyzing data, generating hypotheses, and supporting innovative approaches in various scientific fields. In this work, we investigate how LLMs can handle the transition from conceptual research ideas to well-structured research plans. Effective research planning not only supports scientists in advancing their research but also represents a crucial capability for the development of autonomous research agents. Despite its importance, the field lacks a systematic understanding of LLMs' research planning capability. To rigorously measure this capability, we introduce the Idea2Plan task and Idea2Plan Bench, a benchmark built from 200 ICML 2025 Spotlight and Oral papers released after major LLM training cutoffs. Each benchmark instance includes a research idea and a grading rubric capturing the key components of valid plans. We further propose Idea2Plan JudgeEval, a complementary benchmark to assess the reliability of LLM-based judges against expert annotations. Experimental results show that GPT-5 and GPT-5-mini achieve the strongest performance on the benchmark, though substantial headroom remains for future improvement. Our study provides new insights into LLMs' capability for research planning and lay the groundwork for future progress.

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


### DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving
**Date:** 2025-10-29 | **Arxiv:** [2406.09795](https://arxiv.org/abs/2406.09795)

#### Abstract
The limited availability of high-quality training data poses a major obstacle in data-driven PDE solving, where expensive data collection and resolution constraints severely impact the ability of neural operator networks to learn and generalize the underlying physical system. To address this challenge, we propose DeltaPhi, a novel learning framework that transforms the PDE solving task from learning direct input-output mappings to learning the residuals between similar physical states, a fundamentally different approach to neural operator learning. This reformulation provides implicit data augmentation by exploiting the inherent stability of physical systems where closer initial states lead to closer evolution trajectories. DeltaPhi is architecture-agnostic and can be seamlessly integrated with existing neural operators to enhance their performance. Extensive experiments demonstrate consistent and significant improvements across diverse physical systems including regular and irregular domains, different neural architectures, multiple training data amount, and cross-resolution scenarios, confirming its effectiveness as a general enhancement for neural operators in data-limited PDE solving.

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
* **Construct:** Neural Architecture
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Randomized Neural Network with Adaptive Forward Regularization for Online Task-free Class Incremental Learning
**Date:** 2025-10-27 | **Arxiv:** [2510.21367](https://arxiv.org/abs/2510.21367)

#### Abstract
Class incremental learning (CIL) requires an agent to learn distinct tasks consecutively with knowledge retention against forgetting. Problems impeding the practical applications of CIL methods are twofold: (1) non-i.i.d batch streams and no boundary prompts to update, known as the harsher online task-free CIL (OTCIL) scenario; (2) CIL methods suffer from memory loss in learning long task streams, as shown in Fig. 1 (a). To achieve efficient decision-making and decrease cumulative regrets during the OTCIL process, a randomized neural network (Randomized NN) with forward regularization (-F) is proposed to resist forgetting and enhance learning performance. This general framework integrates unsupervised knowledge into recursive convex optimization, has no learning dissipation, and can outperform the canonical ridge style (-R) in OTCIL. Based on this framework, we derive the algorithm of the ensemble deep random vector functional link network (edRVFL) with adjustable forward regularization (-kF), where k mediates the intensity of the intervention. edRVFL-kF generates one-pass closed-form incremental updates and variable learning rates, effectively avoiding past replay and catastrophic forgetting while achieving superior performance. Moreover, to curb unstable penalties caused by non-i.i.d and mitigate intractable tuning of -kF in OTCIL, we improve it to the plug-and-play edRVFL-kF-Bayes, enabling all hard ks in multiple sub-learners to be self-adaptively determined based on Bayesian learning. Experiments were conducted on 2 image datasets including 6 metrics, dynamic performance, ablation tests, and compatibility, which distinctly validates the efficacy of our OTCIL frameworks with -kF-Bayes and -kF styles.

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


### Unified Implementations of Recurrent Neural Networks in Multiple Deep Learning Frameworks
**Date:** 2025-10-27 | **Arxiv:** [2510.21252](https://arxiv.org/abs/2510.21252)

#### Abstract
Recurrent neural networks (RNNs) are a cornerstone of sequence modeling across various scientific and industrial applications. Owing to their versatility, numerous RNN variants have been proposed over the past decade, aiming to improve the modeling of long-term dependencies and to address challenges such as vanishing and exploding gradients. However, no central library is available to test these variations, and reimplementing diverse architectures can be time-consuming and error-prone, limiting reproducibility and exploration. Here, we introduce three open-source libraries in Julia and Python that centralize numerous recurrent cell implementations and higher-level recurrent architectures. torchrecurrent, RecurrentLayers.jl, and LuxRecurrentLayers.jl offer a consistent framework for constructing and extending RNN models, providing built-in mechanisms for customization and experimentation. All packages are available under the MIT license and actively maintained on GitHub.

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
* **Construct:** Neural Architecture
* **Layer:** Application
* **Limits:** However, no central library is available to test these variations, and reimplementing diverse architectures can be time-consuming and error-prone, limiting reproducibility and exploration.
* **Signal Tags:** #ai

---


### On the Emergence of Linear Analogies in Word Embeddings
**Date:** 2025-10-24 | **Arxiv:** [2505.18651](https://arxiv.org/abs/2505.18651)

#### Abstract
Models such as Word2Vec and GloVe construct word embeddings based on the co-occurrence probability $P(i,j)$ of words $i$ and $j$ in text corpora. The resulting vectors $W_i$ not only group semantically similar words but also exhibit a striking linear analogy structure -- for example, $W_{\text{king}} - W_{\text{man}} + W_{\text{woman}} \approx W_{\text{queen}}$ -- whose theoretical origin remains unclear. Previous observations indicate that this analogy structure: (i) already emerges in the top eigenvectors of the matrix $M(i,j) = P(i,j)/P(i)P(j)$, (ii) strengthens and then saturates as more eigenvectors of $M (i, j)$, which controls the dimension of the embeddings, are included, (iii) is enhanced when using $\log M(i,j)$ rather than $M(i,j)$, and (iv) persists even when all word pairs involved in a specific analogy relation (e.g., king-queen, man-woman) are removed from the corpus. To explain these phenomena, we introduce a theoretical generative model in which words are defined by binary semantic attributes, and co-occurrence probabilities are derived from attribute-based interactions. This model analytically reproduces the emergence of linear analogy structure and naturally accounts for properties (i)-(iv). It can be viewed as giving fine-grained resolution into the role of each additional embedding dimension. It is robust to various forms of noise and agrees well with co-occurrence statistics measured on Wikipedia and the analogy benchmark introduced by Mikolov et al.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards
**Date:** 2025-10-24 | **Arxiv:** [2510.20055](https://arxiv.org/abs/2510.20055)

#### Abstract
Online advertising platforms use automated auctions to connect advertisers with potential customers, requiring effective bidding strategies to maximize profits. Accurate ad impact estimation requires considering three key factors: delayed and long-term effects, cumulative ad impacts such as reinforcement or fatigue, and customer heterogeneity. However, these effects are often not jointly addressed in previous studies. To capture these factors, we model ad bidding as a Contextual Markov Decision Process (CMDP) with delayed Poisson rewards. For efficient estimation, we propose a two-stage maximum likelihood estimator combined with data-splitting strategies, ensuring controlled estimation error based on the first-stage estimator's (in)accuracy. Building on this, we design a reinforcement learning algorithm to derive efficient personalized bidding strategies. This approach achieves a near-optimal regret bound of $\tilde{O}{(dH^2\sqrt{T})}$, where $d$ is the contextual dimension, $H$ is the number of rounds, and $T$ is the number of customers. Our theoretical findings are validated by simulation experiments.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** However, these effects are often not jointly addressed in previous studies.
* **Signal Tags:** #ai

---


### Improving planning and MBRL with temporally-extended actions
**Date:** 2025-10-23 | **Arxiv:** [2505.15754](https://arxiv.org/abs/2505.15754)

#### Abstract
Continuous time systems are often modeled using discrete time dynamics but this requires a small simulation step to maintain accuracy. In turn, this requires a large planning horizon which leads to computationally demanding planning problems and reduced performance. Previous work in model-free reinforcement learning has partially addressed this issue using action repeats where a policy is learned to determine a discrete action duration. Instead we propose to control the continuous decision timescale directly by using temporally-extended actions and letting the planner treat the duration of the action as an additional optimization variable along with the standard action variables. This additional structure has multiple advantages. It speeds up simulation time of trajectories and, importantly, it allows for deep horizon search in terms of primitive actions while using a shallow search depth in the planner. In addition, in the model-based reinforcement learning (MBRL) setting, it reduces compounding errors from model learning and improves training time for models. We show that this idea is effective and that the range for action durations can be automatically selected using a multi-armed bandit formulation and integrated into the MBRL framework. An extensive experimental evaluation both in planning and in MBRL, shows that our approach yields faster planning, better solutions, and that it enables solutions to problems that are not solved in the standard formulation.

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


### R2L: Reliable Reinforcement Learning: Guaranteed Return & Reliable Policies in Reinforcement Learning
**Date:** 2025-10-22 | **Arxiv:** [2510.18074](https://arxiv.org/abs/2510.18074)

#### Abstract
In this work, we address the problem of determining reliable policies in reinforcement learning (RL), with a focus on optimization under uncertainty and the need for performance guarantees. While classical RL algorithms aim at maximizing the expected return, many real-world applications - such as routing, resource allocation, or sequential decision-making under risk - require strategies that ensure not only high average performance but also a guaranteed probability of success. To this end, we propose a novel formulation in which the objective is to maximize the probability that the cumulative return exceeds a prescribed threshold. We demonstrate that this reliable RL problem can be reformulated, via a state-augmented representation, into a standard RL problem, thereby allowing the use of existing RL and deep RL algorithms without the need for entirely new algorithmic frameworks. Theoretical results establish the equivalence of the two formulations and show that reliable strategies can be derived by appropriately adapting well-known methods such as Q-learning or Dueling Double DQN. To illustrate the practical relevance of the approach, we consider the problem of reliable routing, where the goal is not to minimize the expected travel time but rather to maximize the probability of reaching the destination within a given time budget. Numerical experiments confirm that the proposed formulation leads to policies that effectively balance efficiency and reliability, highlighting the potential of reliable RL for applications in stochastic and safety-critical environments.

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


### NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective
**Date:** 2025-10-22 | **Arxiv:** [2510.18258](https://arxiv.org/abs/2510.18258)

#### Abstract
Multi-Task Learning (MTL) enables a single model to learn multiple tasks simultaneously, leveraging knowledge transfer among tasks for enhanced generalization, and has been widely applied across various domains. However, task imbalance remains a major challenge in MTL. Although balancing the convergence speeds of different tasks is an effective approach to address this issue, it is highly challenging to accurately characterize the training dynamics and convergence speeds of multiple tasks within the complex MTL system. To this end, we attempt to analyze the training dynamics in MTL by leveraging Neural Tangent Kernel (NTK) theory and propose a new MTL method, NTKMTL. Specifically, we introduce an extended NTK matrix for MTL and adopt spectral analysis to balance the convergence speeds of multiple tasks, thereby mitigating task imbalance. Based on the approximation via shared representation, we further propose NTKMTL-SR, achieving training efficiency while maintaining competitive performance. Extensive experiments demonstrate that our methods achieve state-of-the-art performance across a wide range of benchmarks, including both multi-task supervised learning and multi-task reinforcement learning. Source code is available at https://github.com/jianke0604/NTKMTL.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** However, task imbalance remains a major challenge in MTL.
* **Signal Tags:** #ai

---


### DFNN: A Deep Fr\'echet Neural Network Framework for Learning Metric-Space-Valued Responses
**Date:** 2025-10-21 | **Arxiv:** [2510.17072](https://arxiv.org/abs/2510.17072)

#### Abstract
Regression with non-Euclidean responses -- e.g., probability distributions, networks, symmetric positive-definite matrices, and compositions -- has become increasingly important in modern applications. In this paper, we propose deep Fréchet neural networks (DFNNs), an end-to-end deep learning framework for predicting non-Euclidean responses -- which are considered as random objects in a metric space -- from Euclidean predictors. Our method leverages the representation-learning power of deep neural networks (DNNs) to the task of approximating conditional Fréchet means of the response given the predictors, the metric-space analogue of conditional expectations, by minimizing a Fréchet risk. The framework is highly flexible, accommodating diverse metrics and high-dimensional predictors. We establish a universal approximation theorem for DFNNs, advancing the state-of-the-art of neural network approximation theory to general metric-space-valued responses without making model assumptions or relying on local smoothing. Empirical studies on synthetic distributional and network-valued responses, as well as a real-world application to predicting employment occupational compositions, demonstrate that DFNNs consistently outperform existing methods.

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
* **Construct:** Neural Architecture
* **Layer:** Application
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Transfer learning strategies for accelerating reinforcement-learning-based flow control
**Date:** 2025-10-21 | **Arxiv:** [2510.16016](https://arxiv.org/abs/2510.16016)

#### Abstract
This work investigates transfer learning strategies to accelerate deep reinforcement learning (DRL) for multifidelity control of chaotic fluid flows. Progressive neural networks (PNNs), a modular architecture designed to preserve and reuse knowledge across tasks, are employed for the first time in the context of DRL-based flow control. In addition, a comprehensive benchmarking of conventional fine-tuning strategies is conducted, evaluating their performance, convergence behavior, and ability to retain transferred knowledge. The Kuramoto-Sivashinsky (KS) system is employed as a benchmark to examine how knowledge encoded in control policies, trained in low-fidelity environments, can be effectively transferred to high-fidelity settings. Systematic evaluations show that while fine-tuning can accelerate convergence, it is highly sensitive to pretraining duration and prone to catastrophic forgetting. In contrast, PNNs enable stable and efficient transfer by preserving prior knowledge and providing consistent performance gains, and are notably robust to overfitting during the pretraining phase. Layer-wise sensitivity analysis further reveals how PNNs dynamically reuse intermediate representations from the source policy while progressively adapting deeper layers to the target task. Moreover, PNNs remain effective even when the source and target environments differ substantially, such as in cases with mismatched physical regimes or control objectives, where fine-tuning strategies often result in suboptimal adaptation or complete failure of knowledge transfer. The results highlight the potential of novel transfer learning frameworks for robust, scalable, and computationally efficient flow control that can potentially be applied to more complex flow configurations.

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
* **Construct:** Neural Architecture
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Understanding Prompt Tuning and In-Context Learning via Meta-Learning
**Date:** 2025-10-21 | **Arxiv:** [2505.17010](https://arxiv.org/abs/2505.17010)

#### Abstract
Prompting is one of the main ways to adapt a pretrained model to target tasks. Besides manually constructing prompts, many prompt optimization methods have been proposed in the literature. Method development is mainly empirically driven, with less emphasis on a conceptual understanding of prompting. In this paper we discuss how optimal prompting can be understood through a Bayesian view, which also implies some fundamental limitations of prompting that can only be overcome by tuning weights. The paper explains in detail how meta-trained neural networks behave as Bayesian predictors over the pretraining distribution, whose hallmark feature is rapid in-context adaptation. Optimal prompting can be studied formally as conditioning these Bayesian predictors, yielding criteria for target tasks where optimal prompting is and is not possible. We support the theory with educational experiments on LSTMs and Transformers, where we compare different versions of prefix-tuning and different weight-tuning methods. We also confirm that soft prefixes, which are sequences of real-valued vectors outside the token alphabet, can lead to very effective prompts for trained and even untrained networks by manipulating activations in ways that are not achievable by hard tokens. This adds an important mechanistic aspect beyond the conceptual Bayesian theory.

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
* **Construct:** Neural Architecture
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### BPL: Bias-adaptive Preference Distillation Learning for Recommender System
**Date:** 2025-10-21 | **Arxiv:** [2510.16076](https://arxiv.org/abs/2510.16076)

#### Abstract
Recommender systems suffer from biases that cause the collected feedback to incompletely reveal user preference. While debiasing learning has been extensively studied, they mostly focused on the specialized (called counterfactual) test environment simulated by random exposure of items, significantly degrading accuracy in the typical (called factual) test environment based on actual user-item interactions. In fact, each test environment highlights the benefit of a different aspect: the counterfactual test emphasizes user satisfaction in the long-terms, while the factual test focuses on predicting subsequent user behaviors on platforms. Therefore, it is desirable to have a model that performs well on both tests rather than only one. In this work, we introduce a new learning framework, called Bias-adaptive Preference distillation Learning (BPL), to gradually uncover user preferences with dual distillation strategies. These distillation strategies are designed to drive high performance in both factual and counterfactual test environments. Employing a specialized form of teacher-student distillation from a biased model, BPL retains accurate preference knowledge aligned with the collected feedback, leading to high performance in the factual test. Furthermore, through self-distillation with reliability filtering, BPL iteratively refines its knowledge throughout the training process. This enables the model to produce more accurate predictions across a broader range of user-item combinations, thereby improving performance in the counterfactual test. Comprehensive experiments validate the effectiveness of BPL in both factual and counterfactual tests. Our implementation is accessible via: https://github.com/SeongKu-Kang/BPL.

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
* **Construct:** General
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### MoRe-ERL: Learning Motion Residuals using Episodic Reinforcement Learning
**Date:** 2025-10-21 | **Arxiv:** [2508.01409](https://arxiv.org/abs/2508.01409)

#### Abstract
We propose MoRe-ERL, a framework that combines Episodic Reinforcement Learning (ERL) and residual learning, which refines preplanned reference trajectories into safe, feasible, and efficient task-specific trajectories. This framework is general enough to incorporate into arbitrary ERL methods and motion generators seamlessly. MoRe-ERL identifies trajectory segments requiring modification while preserving critical task-related maneuvers. Then it generates smooth residual adjustments using B-Spline-based movement primitives to ensure adaptability to dynamic task contexts and smoothness in trajectory refinement. Experimental results demonstrate that residual learning significantly outperforms training from scratch using ERL methods, achieving superior sample efficiency and task performance. Hardware evaluations further validate the framework, showing that policies trained in simulation can be directly deployed in real-world systems, exhibiting a minimal sim-to-real gap.

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
* **Construct:** General
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### LANPO: Bootstrapping Language and Numerical Feedback for Reinforcement Learning in LLMs
**Date:** 2025-10-21 | **Arxiv:** [2510.16552](https://arxiv.org/abs/2510.16552)

#### Abstract
Reinforcement learning in large language models (LLMs) often relies on scalar rewards, a practice that discards valuable textual rationale buried in the rollouts, forcing the model to explore \textit{de novo} with each attempt and hindering sample efficiency. While LLMs can uniquely learn from language feedback provided in-context, naively integrating on-line experiences into RL training presents a paradox: feedback from the same problem risks information leakage and memorization, while feedback from different problems often leads to behavior collapse due to irrelevant context. To resolve this tension, we propose \textbf{Language-And-Numerical Policy Optimization (LANPO)}, a framework that cleanly separates the roles of feedback: language guides exploration, while numerical rewards drive optimization. LANPO builds a dynamic experience pool from past trials and introduces two principles to ensure feedback is effective: \emph{Reward-Agnostic Reflection} for safe intra-sample self-correction and \emph{Relevant Abstraction} to distill generalizable lessons from inter-sample experiences. Across mathematical reasoning benchmarks, LANPO enables 7B and 14B models to significantly outperform strong baselines trained with GRPO in test accuracy. Our work provides a robust method for integrating historical experiences into the LLM RL loop, creating more effective and data-efficient learning agents.

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


### SAMix: Calibrated and Accurate Continual Learning via Sphere-Adaptive Mixup and Neural Collapse
**Date:** 2025-10-20 | **Arxiv:** [2510.15751](https://arxiv.org/abs/2510.15751)

#### Abstract
While most continual learning methods focus on mitigating forgetting and improving accuracy, they often overlook the critical aspect of network calibration, despite its importance. Neural collapse, a phenomenon where last-layer features collapse to their class means, has demonstrated advantages in continual learning by reducing feature-classifier misalignment. Few works aim to improve the calibration of continual models for more reliable predictions. Our work goes a step further by proposing a novel method that not only enhances calibration but also improves performance by reducing overconfidence, mitigating forgetting, and increasing accuracy. We introduce Sphere-Adaptive Mixup (SAMix), an adaptive mixup strategy tailored for neural collapse-based methods. SAMix adapts the mixing process to the geometric properties of feature spaces under neural collapse, ensuring more robust regularization and alignment. Experiments show that SAMix significantly boosts performance, surpassing SOTA methods in continual learning while also improving model calibration. SAMix enhances both across-task accuracy and the broader reliability of predictions, making it a promising advancement for robust continual learning systems.

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


### Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay
**Date:** 2025-10-17 | **Arxiv:** [2507.02901](https://arxiv.org/abs/2507.02901)

#### Abstract
Edge computing scenarios necessitate the development of hardware-efficient online continual learning algorithms to be adaptive to dynamic environment. However, existing algorithms always suffer from high memory overhead and bias towards recently trained tasks. To tackle these issues, this paper proposes a novel online continual learning approach termed as SESLR, which incorporates a sleep enhanced latent replay scheme with spiking neural networks (SNNs). SESLR leverages SNNs' binary spike characteristics to store replay features in single bits, significantly reducing memory overhead. Furthermore, inspired by biological sleep-wake cycles, SESLR introduces a noise-enhanced sleep phase where the model exclusively trains on replay samples with controlled noise injection, effectively mitigating classification bias towards new classes. Extensive experiments on both conventional (MNIST, CIFAR10) and neuromorphic (NMNIST, CIFAR10-DVS) datasets demonstrate SESLR's effectiveness. On Split CIFAR10, SESLR achieves nearly 30% improvement in average accuracy with only one-third of the memory consumption compared to baseline methods. On Split CIFAR10-DVS, it improves accuracy by approximately 10% while reducing memory overhead by a factor of 32. These results validate SESLR as a promising solution for online continual learning in resource-constrained edge computing scenarios.

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
* **Layer:** Hardware
* **Limits:** However, existing algorithms always suffer from high memory overhead and bias towards recently trained tasks.
* **Signal Tags:** #ai

---


### Learning to Undo: Rollback-Augmented Reinforcement Learning with Reversibility Signals
**Date:** 2025-10-17 | **Arxiv:** [2510.14503](https://arxiv.org/abs/2510.14503)

#### Abstract
This paper proposes a reversible learning framework to improve the robustness and efficiency of value based Reinforcement Learning agents, addressing vulnerability to value overestimation and instability in partially irreversible environments. The framework has two complementary core mechanisms: an empirically derived transition reversibility measure called Phi of s and a, and a selective state rollback operation. We introduce an online per state action estimator called Phi that quantifies the likelihood of returning to a prior state within a fixed horizon K. This measure is used to adjust the penalty term during temporal difference updates dynamically, integrating reversibility awareness directly into the value function. The system also includes a selective rollback operator. When an action yields an expected return markedly lower than its instantaneous estimated value and violates a predefined threshold, the agent is penalized and returns to the preceding state rather than progressing. This interrupts sub optimal high risk trajectories and avoids catastrophic steps. By combining reversibility aware evaluation with targeted rollback, the method improves safety, performance, and stability. In the CliffWalking v0 domain, the framework reduced catastrophic falls by over 99.8 percent and yielded a 55 percent increase in mean episode return. In the Taxi v3 domain, it suppressed illegal actions by greater than or equal to 99.9 percent and achieved a 65.7 percent improvement in cumulative reward, while also sharply reducing reward variance in both environments. Ablation studies confirm that the rollback mechanism is the critical component underlying these safety and performance gains, marking a robust step toward safe and reliable sequential decision making.

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


### HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions
**Date:** 2025-10-16 | **Arxiv:** [2510.12733](https://arxiv.org/abs/2510.12733)

#### Abstract
Safe and interpretable motion planning in complex urban environments needs to reason about bidirectional multi-agent interactions. This reasoning requires to estimate the costs of potential ego driving maneuvers. Many existing planners generate initial trajectories with sampling-based methods and refine them by optimizing on learned predictions of future environment states, which requires a cost function that encodes the desired vehicle behavior. Designing such a cost function can be very challenging, especially if a wide range of complex urban scenarios has to be considered. We propose HYPE: HYbrid Planning with Ego proposal-conditioned predictions, a planner that integrates multimodal trajectory proposals from a learned proposal model as heuristic priors into a Monte Carlo Tree Search (MCTS) refinement. To model bidirectional interactions, we introduce an ego-conditioned occupancy prediction model, enabling consistent, scene-aware reasoning. Our design significantly simplifies cost function design in refinement by considering proposal-driven guidance, requiring only minimalistic grid-based cost terms. Evaluations on large-scale real-world benchmarks nuPlan and DeepUrban show that HYPE effectively achieves state-of-the-art performance, especially in safety and adaptability.

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


### Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry
**Date:** 2025-10-15 | **Arxiv:** [2510.09740](https://arxiv.org/abs/2510.09740)

#### Abstract
Active Learning (AL) promises to reduce annotation cost by prioritizing informative samples, yet its reliability is undermined when labels are noisy or when the data distribution shifts. In practice, annotators make mistakes, rare categories are ambiguous, and conventional AL heuristics (uncertainty, diversity) often amplify such errors by repeatedly selecting mislabeled or redundant samples. We propose Reliable Active Learning via Neural Collapse Geometry (NCAL-R), a framework that leverages the emergent geometric regularities of deep networks to counteract unreliable supervision. Our method introduces two complementary signals: (i) a Class-Mean Alignment Perturbation score, which quantifies how candidate samples structurally stabilize or distort inter-class geometry, and (ii) a Feature Fluctuation score, which captures temporal instability of representations across training checkpoints. By combining these signals, NCAL-R prioritizes samples that both preserve class separation and highlight ambiguous regions, mitigating the effect of noisy or redundant labels. Experiments on ImageNet-100 and CIFAR100 show that NCAL-R consistently outperforms standard AL baselines, achieving higher accuracy with fewer labels, improved robustness under synthetic label noise, and stronger generalization to out-of-distribution data. These results suggest that incorporating geometric reliability criteria into acquisition decisions can make Active Learning less brittle to annotation errors and distribution shifts, a key step toward trustworthy deployment in real-world labeling pipelines. Our code is available at https://github.com/Vision-IIITD/NCAL.

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


### Spatial Uncertainty Quantification in Wildfire Forecasting for Climate-Resilient Emergency Planning
**Date:** 2025-10-15 | **Arxiv:** [2510.09666](https://arxiv.org/abs/2510.09666)

#### Abstract
Climate change is intensifying wildfire risks globally, making reliable forecasting critical for adaptation strategies. While machine learning shows promise for wildfire prediction from Earth observation data, current approaches lack uncertainty quantification essential for risk-aware decision making. We present the first systematic analysis of spatial uncertainty in wildfire spread forecasting using multimodal Earth observation inputs. We demonstrate that predictive uncertainty exhibits coherent spatial structure concentrated near fire perimeters. Our novel distance metric reveals high-uncertainty regions form consistent 20-60 meter buffer zones around predicted firelines - directly applicable for emergency planning. Feature attribution identifies vegetation health and fire activity as primary uncertainty drivers. This work enables more robust wildfire management systems supporting communities adapting to increasing fire risk under climate change.

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


### It's 2025 -- Narrative Learning is the new baseline to beat for explainable machine learning
**Date:** 2025-10-15 | **Arxiv:** [2510.09723](https://arxiv.org/abs/2510.09723)

#### Abstract
In this paper, we introduce Narrative Learning, a methodology where models are defined entirely in natural language and iteratively refine their classification criteria using explanatory prompts rather than traditional numerical optimisation. We report on experiments to evaluate the accuracy and potential of this approach using 3 synthetic and 3 natural datasets and compare them against 7 baseline explainable machine learning models. We demonstrate that on 5 out of 6 of these datasets, Narrative Learning became more accurate than the baseline explainable models in 2025 or earlier because of improvements in language models. We also report on trends in the lexicostatistics of these models' outputs as a proxy for the comprehensibility of the explanations.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### How Reinforcement Learning After Next-Token Prediction Facilitates Learning
**Date:** 2025-10-14 | **Arxiv:** [2510.11495](https://arxiv.org/abs/2510.11495)

#### Abstract
Recent advances in reasoning domains with neural networks have primarily been enabled by a training recipe that optimizes Large Language Models, previously trained to predict the next-token in a sequence, with reinforcement learning algorithms. We introduce a framework to study the success of this paradigm, and we theoretically expose the optimization mechanisms by which reinforcement learning improves over next-token prediction in this setting. We study learning from mixture distributions of short and long ``chain-of-thought'' sequences encoding a single task. In particular, when the task consists of predicting the parity of $d$ bits and long sequences are rare, we show how reinforcement learning after next-token prediction enables autoregressive transformers to generalize, whereas mere next-token prediction requires extreme statistical or computational resources to do so. We further explain how reinforcement learning leverages increased test-time computation, manifested in longer responses, to facilitate this learning process. In a simplified setting, we theoretically prove that autoregressive linear models following this training recipe can efficiently learn to predict the parity of $d$ bits as long as the proportion of long demonstrations in the data mix is not exponentially small in the input dimension $d$. Finally, we demonstrate these same phenomena in other settings, including the post-training of Llama-series models on mixture variations of common mathematical reasoning benchmarks.

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
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Little By Little: Continual Learning via Self-Activated Sparse Mixture-of-Rank Adaptive Learning
**Date:** 2025-10-10 | **Arxiv:** [2506.21035](https://arxiv.org/abs/2506.21035)

#### Abstract
Continual learning (CL) with large pre-trained models is challenged by catastrophic forgetting and task interference. Existing LoRA-based Mixture-of-Experts (MoE) approaches mitigate forgetting by assigning and freezing task-specific adapters, but suffer from interference, redundancy, and ambiguous routing due to coarse adapter-level selection. However, this design introduces three key challenges: 1) Interference: Activating full LoRA experts per input leads to subspace interference and prevents selective reuse of useful components across tasks. 2) Redundancy: Newly added experts often duplicate or contradict existing knowledge due to unnecessary activation of unrelated ranks and insufficient reuse of relevant ones. 3) Ambiguity: Overlapping features across tasks confuse the router, resulting in unstable expert assignments. As more experts accumulate, earlier task routing degrades, accelerating forgetting. We propose MoRA, a Mixture-of-Rank Adaptive learning approaches with self-activated and sparse rank activation for CL. Unlike mixing multiple low-rank matrices, MoRA decomposes each rank-r update into r rank-one components, each treated as an independent expert, enabling fine-grained rank-one expert utilization while mitigating interference and redundancy. To avoid ambiguous routing, we propose that each rank-one expert can infer its own relevance via intermediate activations. Coupled with our proposed rank pruning and activation budgets, MoRA adaptively selects a sparse mixture of ranks per input. We validate MoRA on continual learning benchmarks using CLIP and language models, analyzing both in-domain learning and out-of-domain forgetting/generalization during fine-tuning. MoRA shows significant effectiveness in enhancing CL with PTMs, and improving generalization while mitigating forgetting.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** However, this design introduces three key challenges: 1) Interference: Activating full LoRA experts per input leads to subspace interference and prevents selective reuse of useful components across tasks.
* **Signal Tags:** #ai

---


### xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning
**Date:** 2025-10-10 | **Arxiv:** [2510.08439](https://arxiv.org/abs/2510.08439)

#### Abstract
Modern LLM deployments confront a widening cost-performance spectrum: premium models deliver strong reasoning but are expensive, while lightweight models are economical yet brittle on complex tasks. Static escalation rules and keyword heuristics under-utilize this spectrum and fail to adapt across task types. We present xRouter, a tool-calling-based routing system in which a learned router can either answer directly or invoke one or more external models. The router is trained end-to-end with reinforcement learning using an explicit, cost-aware reward that encodes cost-performance trade-offs, eliminating the need for hand-engineered routing rules. Our implementation encompasses the full reinforcement learning framework, including reward and cost accounting, as well as the deployment and evaluation pipelines. Across diverse benchmarks, xRouter achieves strong cost-performance trade-offs (e.g., substantial cost reductions at comparable task completion rates), and provides empirical insights into what reliably helps learned routing and what does not, ranging from model trainability to the difficulty of eliciting sophisticated orchestration behaviors in small open models. We hope these findings and our open implementation will serve as a practical substrate for advancing learned, cost-aware LLM orchestration.

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
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning
**Date:** 2025-10-10 | **Arxiv:** [2502.03752](https://arxiv.org/abs/2502.03752)

#### Abstract
Meta-reinforcement learning (Meta-RL) facilitates rapid adaptation to unseen tasks but faces challenges in long-horizon environments. Skill-based approaches tackle this by decomposing state-action sequences into reusable skills and employing hierarchical decision-making. However, these methods are highly susceptible to noisy offline demonstrations, leading to unstable skill learning and degraded performance. To address this, we propose Self-Improving Skill Learning (SISL), which performs self-guided skill refinement using decoupled high-level and skill improvement policies, while applying skill prioritization via maximum return relabeling to focus updates on task-relevant trajectories, resulting in robust and stable adaptation even under noisy and suboptimal data. By mitigating the effect of noise, SISL achieves reliable skill learning and consistently outperforms other skill-based meta-RL methods on diverse long-horizon tasks.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** However, these methods are highly susceptible to noisy offline demonstrations, leading to unstable skill learning and degraded performance.
* **Signal Tags:** #ai

---


### AbsoluteNet: A Deep Learning Neural Network to Classify Cerebral Hemodynamic Responses of Auditory Processing
**Date:** 2025-10-09 | **Arxiv:** [2506.00039](https://arxiv.org/abs/2506.00039)

#### Abstract
In recent years, deep learning (DL) approaches have demonstrated promising results in decoding hemodynamic responses captured by functional near-infrared spectroscopy (fNIRS), particularly in the context of brain-computer interface (BCI) applications. This work introduces AbsoluteNet, a novel deep learning architecture designed to classify auditory event-related responses recorded using fNIRS. The proposed network is built upon principles of spatio-temporal convolution and customized activation functions. Our model was compared against several models, namely fNIRSNET, MDNN, DeepConvNet, and ShallowConvNet. The results showed that AbsoluteNet outperforms existing models, reaching 87.0% accuracy, 84.8% sensitivity, and 89.2% specificity in binary classification, surpassing fNIRSNET, the second-best model, by 3.8% in accuracy. These findings underscore the effectiveness of our proposed deep learning model in decoding hemodynamic responses related to auditory processing and highlight the importance of spatio-temporal feature aggregation and customized activation functions to better fit fNIRS dynamics.

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
* **Construct:** Neural Architecture
* **Layer:** Application
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Learning Semantics, Not Addresses: Runtime Neural Prefetching for Far Memory
**Date:** 2025-10-07 | **Arxiv:** [2506.00384](https://arxiv.org/abs/2506.00384)

#### Abstract
Memory prefetching has long boosted CPU caches and is increasingly vital for far-memory systems, where large portions of memory are offloaded to cheaper, remote tiers. While effective prefetching requires accurate prediction of future accesses, prior ML approaches have been limited to simulation or small-scale hardware. We introduce FarSight, the first Linux-based far-memory system to leverage deep learning by decoupling application semantics from runtime memory layout. This separation enables offline-trained models to predict access patterns over a compact ordinal vocabulary, which are resolved at runtime through lightweight mappings. Across four data-intensive workloads, FarSight delivers up to 3.6x higher performance than the state-of-the-art.

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
* **Construct:** General
* **Layer:** Application
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### In-Context Compositional Q-Learning for Offline Reinforcement Learning
**Date:** 2025-09-30 | **Arxiv:** [2509.24067](https://arxiv.org/abs/2509.24067)

#### Abstract
Accurately estimating the Q-function is a central challenge in offline reinforcement learning. However, existing approaches often rely on a single global Q-function, which struggles to capture the compositional nature of tasks involving diverse subtasks. We propose In-context Compositional Q-Learning (\texttt{ICQL}), the first offline RL framework that formulates Q-learning as a contextual inference problem, using linear Transformers to adaptively infer local Q-functions from retrieved transitions without explicit subtask labels. Theoretically, we show that under two assumptions--linear approximability of the local Q-function and accurate weight inference from retrieved context--\texttt{ICQL} achieves bounded Q-function approximation error, and supports near-optimal policy extraction. Empirically, \texttt{ICQL} substantially improves performance in offline settings: improving performance in kitchen tasks by up to 16.4\%, and in Gym and Adroit tasks by up to 8.6\% and 6.3\%. These results highlight the underexplored potential of in-context learning for robust and compositional value estimation, positioning \texttt{ICQL} as a principled and effective framework for offline RL.

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
* **Layer:** Infrastructure
* **Limits:** However, existing approaches often rely on a single global Q-function, which struggles to capture the compositional nature of tasks involving diverse subtasks.
* **Signal Tags:** #ai

---


### A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers
**Date:** 2025-09-30 | **Arxiv:** [2502.01310](https://arxiv.org/abs/2502.01310)

#### Abstract
Neural network-based optimal transport (OT) is a recent and fruitful direction in the generative modeling community. It finds its applications in various fields such as domain translation, image super-resolution, computational biology and others. Among the existing OT approaches, of considerable interest are adversarial minimax solvers based on semi-dual formulations of OT problems. While promising, these methods lack theoretical investigation from a statistical learning perspective. Our work fills this gap by establishing upper bounds on the generalization error of an approximate OT map recovered by the minimax quadratic OT solver. Importantly, the bounds we derive depend solely on some standard statistical and mathematical properties of the considered functional classes (neural nets). While our analysis focuses on the quadratic OT, we believe that similar bounds could be derived for general OT case, paving the promising direction for future research.

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


### Policy Compatible Skill Incremental Learning via Lazy Learning Interface
**Date:** 2025-09-26 | **Arxiv:** [2509.20612](https://arxiv.org/abs/2509.20612)

#### Abstract
Skill Incremental Learning (SIL) is the process by which an embodied agent expands and refines its skill set over time by leveraging experience gained through interaction with its environment or by the integration of additional data. SIL facilitates efficient acquisition of hierarchical policies grounded in reusable skills for downstream tasks. However, as the skill repertoire evolves, it can disrupt compatibility with existing skill-based policies, limiting their reusability and generalization. In this work, we propose SIL-C, a novel framework that ensures skill-policy compatibility, allowing improvements in incrementally learned skills to enhance the performance of downstream policies without requiring policy re-training or structural adaptation. SIL-C employs a bilateral lazy learning-based mapping technique to dynamically align the subtask space referenced by policies with the skill space decoded into agent behaviors. This enables each subtask, derived from the policy's decomposition of a complex task, to be executed by selecting an appropriate skill based on trajectory distribution similarity. We evaluate SIL-C across diverse SIL scenarios and demonstrate that it maintains compatibility between evolving skills and downstream policies while ensuring efficiency throughout the learning process.

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
* **Limits:** However, as the skill repertoire evolves, it can disrupt compatibility with existing skill-based policies, limiting their reusability and generalization.
* **Signal Tags:** #ai

---


### Adaptive Approach to Enhance Machine Learning Scheduling Algorithms During Runtime Using Reinforcement Learning in Metascheduling Applications
**Date:** 2025-09-26 | **Arxiv:** [2509.20520](https://arxiv.org/abs/2509.20520)

#### Abstract
Metascheduling in time-triggered architectures has been crucial in adapting to dynamic and unpredictable environments, ensuring the reliability and efficiency of task execution. However, traditional approaches face significant challenges when training Artificial Intelligence (AI) scheduling inferences offline, particularly due to the complexities involved in constructing a comprehensive Multi-Schedule Graph (MSG) that accounts for all possible scenarios. The process of generating an MSG that captures the vast probability space, especially when considering context events like hardware failures, slack variations, or mode changes, is resource-intensive and often infeasible. To address these challenges, we propose an adaptive online learning unit integrated within the metascheduler to enhance performance in real-time. The primary motivation for developing this unit stems from the limitations of offline training, where the MSG created is inherently a subset of the complete space, focusing only on the most probable and critical context events. In the online mode, Reinforcement Learning (RL) plays a pivotal role by continuously exploring and discovering new scheduling solutions, thus expanding the MSG and enhancing system performance over time. This dynamic adaptation allows the system to handle unexpected events and complex scheduling scenarios more effectively. Several RL models were implemented within the online learning unit, each designed to address specific challenges in scheduling. These models not only facilitate the discovery of new solutions but also optimize existing schedulers, particularly when stricter deadlines or new performance criteria are introduced. By continuously refining the AI inferences through real-time training, the system remains flexible and capable of meeting evolving demands, thus ensuring robustness and efficiency in large-scale, safety-critical environments.

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
* **Limits:** However, traditional approaches face significant challenges when training Artificial Intelligence (AI) scheduling inferences offline, particularly due to the complexities involved in constructing a comprehensive Multi-Schedule Graph (MSG) that accounts for all possible scenarios.
* **Signal Tags:** #ai

---


### A Variational Framework for Residual-Based Adaptivity in Neural PDE Solvers and Operator Learning
**Date:** 2025-09-18 | **Arxiv:** [2509.14198](https://arxiv.org/abs/2509.14198)

#### Abstract
Residual-based adaptive strategies are widely used in scientific machine learning but remain largely heuristic. We introduce a unifying variational framework that formalizes these methods by integrating convex transformations of the residual. Different transformations correspond to distinct objective functionals: exponential weights target the minimization of uniform error, while linear weights recover the minimization of quadratic error. Within this perspective, adaptive weighting is equivalent to selecting sampling distributions that optimize the primal objective, thereby linking discretization choices directly to error metrics. This principled approach yields three benefits: (1) it enables systematic design of adaptive schemes across norms, (2) reduces discretization error through variance reduction of the loss estimator, and (3) enhances learning dynamics by improving the gradient signal-to-noise ratio. Extending the framework to operator learning, we demonstrate substantial performance gains across optimizers and architectures. Our results provide a theoretical justification of residual-based adaptivity and establish a foundation for principled discretization and training strategies.

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
* **Construct:** Neural Architecture
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Decoupling Search and Learning in Neural Net Training
**Date:** 2025-09-16 | **Arxiv:** [2509.10973](https://arxiv.org/abs/2509.10973)

#### Abstract
Gradient descent typically converges to a single minimum of the training loss without mechanisms to explore alternative minima that may generalize better. Searching for diverse minima directly in high-dimensional parameter space is generally intractable. To address this, we propose a framework that performs training in two distinct phases: search in a tractable representation space (the space of intermediate activations) to find diverse representational solutions, and gradient-based learning in parameter space by regressing to those searched representations. Through evolutionary search, we discover representational solutions whose fitness and diversity scale with compute--larger populations and more generations produce better and more varied solutions. These representations prove to be learnable: networks trained by regressing to searched representations approach SGD's performance on MNIST, CIFAR-10, and CIFAR-100. Performance improves with search compute up to saturation. The resulting models differ qualitatively from networks trained with gradient descent, following different representational trajectories during training. This work demonstrates how future training algorithms could overcome gradient descent's exploratory limitations by decoupling search in representation space from efficient gradient-based learning in parameter space.

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
* **Construct:** Neural Architecture
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Framing AI System Benchmarking as a Learning Task: FlexBench and the Open MLPerf Dataset
**Date:** 2025-09-16 | **Arxiv:** [2509.11413](https://arxiv.org/abs/2509.11413)

#### Abstract
Existing AI system benchmarks such as MLPerf often struggle to keep pace with the rapidly evolving AI landscape, making it difficult to support informed deployment, optimization, and co-design decisions for AI systems. We suggest that benchmarking itself can be framed as an AI task - one in which models are continuously evaluated and optimized across diverse datasets, software, and hardware, using key metrics such as accuracy, latency, throughput, energy consumption, and cost. To support this perspective, we present FlexBench: a modular extension of the MLPerf LLM inference benchmark, integrated with HuggingFace and designed to provide relevant and actionable insights. Benchmarking results and metadata are collected into an Open MLPerf Dataset, which can be collaboratively curated, extended, and leveraged for predictive modeling and feature engineering. We successfully validated the FlexBench concept through MLPerf Inference submissions, including evaluations of DeepSeek R1 and LLaMA 3.3 on commodity servers. The broader objective is to enable practitioners to make cost-effective AI deployment decisions that reflect their available resources, requirements, and constraints.

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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining
**Date:** 2025-09-08 | **Arxiv:** [2509.05291](https://arxiv.org/abs/2509.05291)

#### Abstract
Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining.

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
* **Layer:** Theory
* **Limits:** However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities.
* **Signal Tags:** #ai

---


### From Federated Learning to X-Learning: Breaking the Barriers of Decentrality Through Random Walks
**Date:** 2025-09-05 | **Arxiv:** [2509.03709](https://arxiv.org/abs/2509.03709)

#### Abstract
We provide our perspective on X-Learning (XL), a novel distributed learning architecture that generalizes and extends the concept of decentralization. Our goal is to present a vision for XL, introducing its unexplored design considerations and degrees of freedom. To this end, we shed light on the intuitive yet non-trivial connections between XL, graph theory, and Markov chains. We also present a series of open research directions to stimulate further research.

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
* **Construct:** Neural Architecture
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Ultra Strong Machine Learning: Teaching Humans Active Learning Strategies via Automated AI Explanations
**Date:** 2025-09-03 | **Arxiv:** [2509.00961](https://arxiv.org/abs/2509.00961)

#### Abstract
Ultra Strong Machine Learning (USML) refers to symbolic learning systems that not only improve their own performance but can also teach their acquired knowledge to quantifiably improve human performance. In this work, we present LENS (Logic Programming Explanation via Neural Summarisation), a neuro-symbolic method that combines symbolic program synthesis with large language models (LLMs) to automate the explanation of machine-learned logic programs in natural language. LENS addresses a key limitation of prior USML approaches by replacing hand-crafted explanation templates with scalable automated generation. Through systematic evaluation using multiple LLM judges and human validation, we demonstrate that LENS generates superior explanations compared to direct LLM prompting and hand-crafted templates. To investigate whether LENS can teach transferable active learning strategies, we carried out a human learning experiment across three related domains. Our results show no significant human performance improvements, suggesting that comprehensive LLM responses may overwhelm users for simpler problems rather than providing learning support. Our work provides a solid foundation for building effective USML systems to support human learning. The source code is available on: https://github.com/lun-ai/LENS.git.

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
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### A Comparative Analysis of Reinforcement Learning and Conventional Deep Learning Approaches for Bearing Fault Diagnosis
**Date:** 2025-09-03 | **Arxiv:** [2506.19929](https://arxiv.org/abs/2506.19929)

#### Abstract
Bearing faults in rotating machinery can lead to significant operational disruptions and maintenance costs. Modern methods for bearing fault diagnosis rely heavily on vibration analysis and machine learning techniques, which often require extensive labeled data and may not adapt well to dynamic environments. This study explores the feasibility of reinforcement learning (RL), specifically Deep Q-Networks (DQNs), for bearing fault classification tasks in machine condition monitoring to enhance the accuracy and adaptability of bearing fault diagnosis. The results demonstrate that while RL models developed in this study can match the performance of traditional supervised learning models under controlled conditions, they excel in adaptability when equipped with optimized reward structures. However, their computational demands highlight areas for further improvement. These findings demonstrate RL's potential to complement traditional methods, paving the way for adaptive diagnostic frameworks.

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
* **Construct:** Neural Architecture
* **Layer:** Infrastructure
* **Limits:** However, their computational demands highlight areas for further improvement.
* **Signal Tags:** #ai

---


### A Closer Look at Adversarial Suffix Learning for Jailbreaking LLMs: Augmented Adversarial Trigger Learning
**Date:** 2025-08-20 | **Arxiv:** [2503.12339](https://arxiv.org/abs/2503.12339)

#### Abstract
Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning

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


### Results of the NeurIPS 2023 Neural MMO Competition on Multi-task Reinforcement Learning
**Date:** 2025-08-19 | **Arxiv:** [2508.12524](https://arxiv.org/abs/2508.12524)

#### Abstract
We present the results of the NeurIPS 2023 Neural MMO Competition, which attracted over 200 participants and submissions. Participants trained goal-conditional policies that generalize to tasks, maps, and opponents never seen during training. The top solution achieved a score 4x higher than our baseline within 8 hours of training on a single 4090 GPU. We open-source everything relating to Neural MMO and the competition under the MIT license, including the policy weights and training code for our baseline and for the top submissions.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Memorisation and forgetting in a learning Hopfield neural network: bifurcation mechanisms, attractors and basins
**Date:** 2025-08-15 | **Arxiv:** [2508.10765](https://arxiv.org/abs/2508.10765)

#### Abstract
Despite explosive expansion of artificial intelligence based on artificial neural networks (ANNs), these are employed as "black boxes'', as it is unclear how, during learning, they form memories or develop unwanted features, including spurious memories and catastrophic forgetting. Much research is available on isolated aspects of learning ANNs, but due to their high dimensionality and non-linearity, their comprehensive analysis remains a challenge. In ANNs, knowledge is thought to reside in connection weights or in attractor basins, but these two paradigms are not linked explicitly. Here we comprehensively analyse mechanisms of memory formation in an 81-neuron Hopfield network undergoing Hebbian learning by revealing bifurcations leading to formation and destruction of attractors and their basin boundaries. We show that, by affecting evolution of connection weights, the applied stimuli induce a pitchfork and then a cascade of saddle-node bifurcations creating new attractors with their basins that can code true or spurious memories, and an abrupt disappearance of old memories (catastrophic forgetting). With successful learning, new categories are represented by the basins of newly born point attractors, and their boundaries by the stable manifolds of new saddles. With this, memorisation and forgetting represent two manifestations of the same mechanism. Our strategy to analyse high-dimensional learning ANNs is universal and applicable to recurrent ANNs of any form. The demonstrated mechanisms of memory formation and of catastrophic forgetting shed light on the operation of a wider class of recurrent ANNs and could aid the development of approaches to mitigate their flaws.

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
* **Construct:** Neural Architecture
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG Classification with Contrastive Learning
**Date:** 2025-08-13 | **Arxiv:** [2504.08713](https://arxiv.org/abs/2504.08713)

#### Abstract
Deep learning-based electrocardiogram (ECG) classification has shown impressive performance but clinical adoption has been slowed by the lack of transparent and faithful explanations. Post hoc methods such as saliency maps may fail to reflect a model's true decision process. Prototype-based reasoning offers a more transparent alternative by grounding decisions in similarity to learned representations of real ECG segments, enabling faithful, case-based explanations. We introduce ProtoECGNet, a prototype-based deep learning model for interpretable, multi-label ECG classification. ProtoECGNet employs a structured, multi-branch architecture that reflects clinical interpretation workflows: it integrates a 1D CNN with global prototypes for rhythm classification, a 2D CNN with time-localized prototypes for morphology-based reasoning, and a 2D CNN with global prototypes for diffuse abnormalities. Each branch is trained with a prototype loss designed for multi-label learning, combining clustering, separation, diversity, and a novel contrastive loss that encourages appropriate separation between prototypes of unrelated classes while allowing clustering for frequently co-occurring diagnoses. We evaluate ProtoECGNet on all 71 diagnostic labels from the PTB-XL dataset, demonstrating competitive performance relative to state-of-the-art black-box models while providing structured, case-based explanations. To assess prototype quality, we conduct a structured clinician review of the final model's projected prototypes, finding that they are rated as representative and clear. ProtoECGNet shows that prototype learning can be effectively scaled to complex, multi-label time-series classification, offering a practical path toward transparent and trustworthy deep learning models for clinical decision support.

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
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---
