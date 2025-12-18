# Vol 18 Graph   Topology
*Enriched by BITCOREOS | Phase 4 Batch 4*

---

### MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver
**Date:** 2025-10-31 | **Arxiv:** [2506.02935](https://hub.bitwiki.org/t/mtl-kd-multi-task-learning-via-knowledge-distillation-for-generalizable-neural-vehicle-routing-solver/20701)

#### Abstract
Multi-Task Learning (MTL) in Neural Combinatorial Optimization (NCO) is a promising approach to train a unified model capable of solving multiple Vehicle Routing Problem (VRP) variants. However, existing Reinforcement Learning (RL)-based multi-task methods can only train light decoder models on small-scale problems, exhibiting limited generalization ability when solving large-scale problems. To overcome this limitation, this work introduces a novel multi-task learning method driven by knowledge distillation (MTL-KD), which enables the efficient training of heavy decoder models with strong generalization ability. The proposed MTL-KD method transfers policy knowledge from multiple distinct RL-based single-task models to a single heavy decoder model, facilitating label-free training and effectively improving the model's generalization ability across diverse tasks. In addition, we introduce a flexible inference strategy termed Random Reordering Re-Construction (R3C), which is specifically adapted for diverse VRP tasks and further boosts the performance of the multi-task model. Experimental results on 6 seen and 10 unseen VRP variants with up to 1000 nodes indicate that our proposed method consistently achieves superior performance on both uniform and real-world benchmarks, demonstrating robust generalization abilities.

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
* **Limits:** However, existing Reinforcement Learning (RL)-based multi-task methods can only train light decoder models on small-scale problems, exhibiting limited generalization ability when solving large-scale problems.
* **Signal Tags:** #ai

---


### Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Statistical Guarantees
**Date:** 2025-10-30 | **Arxiv:** [2510.24754](https://hub.bitwiki.org/t/certainty-in-uncertainty-reasoning-over-uncertain-knowledge-graphs-with-statistical-guarantees/20231)

#### Abstract
Uncertain knowledge graph embedding (UnKGE) methods learn vector representations that capture both structural and uncertainty information to predict scores of unseen triples. However, existing methods produce only point estimates, without quantifying predictive uncertainty-limiting their reliability in high-stakes applications where understanding confidence in predictions is crucial. To address this limitation, we propose \textsc{UnKGCP}, a framework that generates prediction intervals guaranteed to contain the true score with a user-specified level of confidence. The length of the intervals reflects the model's predictive uncertainty. \textsc{UnKGCP} builds on the conformal prediction framework but introduces a novel nonconformity measure tailored to UnKGE methods and an efficient procedure for interval construction. We provide theoretical guarantees for the intervals and empirically verify these guarantees. Extensive experiments on standard benchmarks across diverse UnKGE methods further demonstrate that the intervals are sharp and effectively capture predictive uncertainty.

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
* **Limits:** However, existing methods produce only point estimates, without quantifying predictive uncertainty-limiting their reliability in high-stakes applications where understanding confidence in predictions is crucial.
* **Signal Tags:** #ai

---


### Quantifying Distributional Invariance in Causal Subgraph for IRM-Free Graph Generalization
**Date:** 2025-10-24 | **Arxiv:** [2510.20295](https://hub.bitwiki.org/t/quantifying-distributional-invariance-in-causal-subgraph-for-irm-free-graph-generalization/19144)

#### Abstract
Out-of-distribution generalization under distributional shifts remains a critical challenge for graph neural networks. Existing methods generally adopt the Invariant Risk Minimization (IRM) framework, requiring costly environment annotations or heuristically generated synthetic splits. To circumvent these limitations, in this work, we aim to develop an IRM-free method for capturing causal subgraphs. We first identify that causal subgraphs exhibit substantially smaller distributional variations than non-causal components across diverse environments, which we formalize as the Invariant Distribution Criterion and theoretically prove in this paper. Building on this criterion, we systematically uncover the quantitative relationship between distributional shift and representation norm for identifying the causal subgraph, and investigate its underlying mechanisms in depth. Finally, we propose an IRM-free method by introducing a norm-guided invariant distribution objective for causal subgraph discovery and prediction. Extensive experiments on two widely used benchmarks demonstrate that our method consistently outperforms state-of-the-art methods in graph generalization.

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


### DAG-Math: Graph-Guided Mathematical Reasoning in LLMs
**Date:** 2025-10-24 | **Arxiv:** [2510.19842](https://hub.bitwiki.org/t/dag-math-graph-guided-mathematical-reasoning-in-llms/19219)

#### Abstract
Large Language Models (LLMs) demonstrate strong performance on mathematical problems when prompted with Chain-of-Thought (CoT), yet it remains unclear whether this success stems from search, rote procedures, or rule-consistent reasoning. To address this, we propose modeling CoT as a certain rule-based stochastic process over directed acyclic graphs (DAGs), where nodes represent intermediate derivation states and edges encode rule applications. Within this framework, we introduce logical closeness, a metric that quantifies how well a model's CoT trajectory (i.e., the LLM's final output) adheres to the DAG structure, providing evaluation beyond classical PASS@k metrics. Building on this, we introduce the DAG-MATH CoT format and construct a benchmark that guides LLMs to generate CoT trajectories in this format, thereby enabling the evaluation of their reasoning ability under our framework. Across standard mathematical reasoning datasets, our analysis uncovers statistically significant differences in reasoning fidelity among representative LLM families-even when PASS@k is comparable-highlighting gaps between final-answer accuracy and rule-consistent derivation. Our framework provides a balance between free-form CoT and formal proofs systems, offering actionable diagnostics for LLMs reasoning evaluation. Our benchmark and code are available at: https://github.com/YuanheZ/DAG-MATH-Formatted-CoT.

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


### Enhancing Graph Neural Networks: A Mutual Learning Approach
**Date:** 2025-10-23 | **Arxiv:** [2510.19223](https://hub.bitwiki.org/t/enhancing-graph-neural-networks-a-mutual-learning-approach/18858)

#### Abstract
Knowledge distillation (KD) techniques have emerged as a powerful tool for transferring expertise from complex teacher models to lightweight student models, particularly beneficial for deploying high-performance models in resource-constrained devices. This approach has been successfully applied to graph neural networks (GNNs), harnessing their expressive capabilities to generate node embeddings that capture structural and feature-related information. In this study, we depart from the conventional KD approach by exploring the potential of collaborative learning among GNNs. In the absence of a pre-trained teacher model, we show that relatively simple and shallow GNN architectures can synergetically learn efficient models capable of performing better during inference, particularly in tackling multiple tasks. We propose a collaborative learning framework where ensembles of student GNNs mutually teach each other throughout the training process. We introduce an adaptive logit weighting unit to facilitate efficient knowledge exchange among models and an entropy enhancement technique to improve mutual learning. These components dynamically empower the models to adapt their learning strategies during training, optimizing their performance for downstream tasks. Extensive experiments conducted on three datasets each for node and graph classification demonstrate the effectiveness of our approach.

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


### HUMAP: Hierarchical Uniform Manifold Approximation and Projection
**Date:** 2025-10-21 | **Arxiv:** [2106.07718](https://hub.bitwiki.org/t/humap-hierarchical-uniform-manifold-approximation-and-projection/18361)

#### Abstract
Dimensionality reduction (DR) techniques help analysts to understand patterns in high-dimensional spaces. These techniques, often represented by scatter plots, are employed in diverse science domains and facilitate similarity analysis among clusters and data samples. For datasets containing many granularities or when analysis follows the information visualization mantra, hierarchical DR techniques are the most suitable approach since they present major structures beforehand and details on demand. This work presents HUMAP, a novel hierarchical dimensionality reduction technique designed to be flexible on preserving local and global structures and preserve the mental map throughout hierarchical exploration. We provide empirical evidence of our technique's superiority compared with current hierarchical approaches and show a case study applying HUMAP for dataset labelling.

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


### HySim-LLM: Embedding-Weighted Fine-Tuning Bounds and Manifold Denoising for Domain-Adapted LLMs
**Date:** 2025-10-10 | **Arxiv:** [2510.07796](https://hub.bitwiki.org/t/hysim-llm-embedding-weighted-fine-tuning-bounds-and-manifold-denoising-for-domain-adapted-llms/15951)

#### Abstract
The extraction and standardization of pharmacokinetic (PK) information from scientific literature remain significant challenges in computational pharmacology, which limits the reliability of data-driven models in drug development. Large language models (LLMs) have achieved remarkable progress in text understanding and reasoning, yet their adaptation to structured biomedical data, such as PK tables, remains constrained by heterogeneity, noise, and domain shift. To address these limitations, we propose HySim-LLM, a unified mathematical and computational framework that integrates embedding-weighted fine-tuning and manifold-aware denoising to enhance the robustness and interpretability of LLMs. We establish two theoretical results: (1) a similarity-weighted generalization bound that quantifies adaptation performance under embedding divergence, and (2) a manifold-based denoising guarantee that bounds loss contributions from noisy or off-manifold samples. These theorems provide a principled foundation for fine-tuning LLMs in structured biomedical settings. The framework offers a mathematically grounded pathway toward reliable and interpretable LLM adaptation for biomedical and data-intensive scientific domains.

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


### SAFA-SNN: Sparsity-Aware On-Device Few-Shot Class-Incremental Learning with Fast-Adaptive Structure of Spiking Neural Network
**Date:** 2025-10-07 | **Arxiv:** [2510.03648](https://hub.bitwiki.org/t/safa-snn-sparsity-aware-on-device-few-shot-class-incremental-learning-with-fast-adaptive-structure-of-spiking-neural-network/14829)

#### Abstract
Continuous learning of novel classes is crucial for edge devices to preserve data privacy and maintain reliable performance in dynamic environments. However, the scenario becomes particularly challenging when data samples are insufficient, requiring on-device few-shot class-incremental learning (FSCIL) to maintain consistent model performance. Although existing work has explored parameter-efficient FSCIL frameworks based on artificial neural networks (ANNs), their deployment is still fundamentally constrained by limited device resources. Inspired by neural mechanisms, Spiking neural networks (SNNs) process spatiotemporal information efficiently, offering lower energy consumption, greater biological plausibility, and compatibility with neuromorphic hardware than ANNs. In this work, we present an SNN-based method for On-Device FSCIL, i.e., Sparsity-Aware and Fast Adaptive SNN (SAFA-SNN). We first propose sparsity-conditioned neuronal dynamics, in which most neurons remain stable while a subset stays active, thereby mitigating catastrophic forgetting. To further cope with spike non-differentiability in gradient estimation, we employ zeroth-order optimization. Moreover, during incremental learning sessions, we enhance the discriminability of new classes through subspace projection, which alleviates overfitting to novel classes. Extensive experiments conducted on two standard benchmark datasets (CIFAR100 and Mini-ImageNet) and three neuromorphic datasets (CIFAR-10-DVS, DVS128gesture, and N-Caltech101) demonstrate that SAFA-SNN outperforms baseline methods, specifically achieving at least 4.01% improvement at the last incremental session on Mini-ImageNet and 20% lower energy cost over baseline methods with practical implementation.

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
* **Limits:** However, the scenario becomes particularly challenging when data samples are insufficient, requiring on-device few-shot class-incremental learning (FSCIL) to maintain consistent model performance.
* **Signal Tags:** #ai

---


### The causal structure of galactic astrophysics
**Date:** 2025-10-02 | **Arxiv:** [2510.01112](https://hub.bitwiki.org/t/the-causal-structure-of-galactic-astrophysics/13919)

#### Abstract
Data-driven astrophysics currently relies on the detection and characterisation of correlations between objects' properties, which are then used to test physical theories that make predictions for them. This process fails to utilise information in the data that forms a crucial part of the theories' predictions, namely which variables are directly correlated (as opposed to accidentally correlated through others), the directions of these determinations, and the presence or absence of confounders that correlate variables in the dataset but are themselves absent from it. We propose to recover this information through causal discovery, a well-developed methodology for inferring the causal structure of datasets that is however almost entirely unknown to astrophysics. We develop a causal discovery algorithm suitable for large astrophysical datasets and illustrate it on $\sim$5$\times10^5$ low-redshift galaxies from the Nasa Sloan Atlas, demonstrating its ability to distinguish physical mechanisms that are degenerate on the basis of correlations alone.

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


### Learning Dynamic Graph Embeddings with Neural Controlled Differential Equations
**Date:** 2025-10-02 | **Arxiv:** [2302.11354](https://hub.bitwiki.org/t/learning-dynamic-graph-embeddings-with-neural-controlled-differential-equations/13925)

#### Abstract
This paper focuses on representation learning for dynamic graphs with temporal interactions. A fundamental issue is that both the graph structure and the nodes own their own dynamics, and their blending induces intractable complexity in the temporal evolution over graphs. Drawing inspiration from the recent progress of physical dynamic models in deep neural networks, we propose Graph Neural Controlled Differential Equations (GN-CDEs), a continuous-time framework that jointly models node embeddings and structural dynamics by incorporating a graph enhanced neural network vector field with a time-varying graph path as the control signal. Our framework exhibits several desirable characteristics, including the ability to express dynamics on evolving graphs without piecewise integration, the capability to calibrate trajectories with subsequent data, and robustness to missing observations. Empirical evaluation on a range of dynamic graph representation learning tasks demonstrates the effectiveness of our proposed approach in capturing the complex dynamics of dynamic graphs.

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


### Guaranteed Noisy CP Tensor Recovery via Riemannian Optimization on the Segre Manifold
**Date:** 2025-10-02 | **Arxiv:** [2510.00569](https://hub.bitwiki.org/t/guaranteed-noisy-cp-tensor-recovery-via-riemannian-optimization-on-the-segre-manifold/13678)

#### Abstract
Recovering a low-CP-rank tensor from noisy linear measurements is a central challenge in high-dimensional data analysis, with applications spanning tensor PCA, tensor regression, and beyond. We exploit the intrinsic geometry of rank-one tensors by casting the recovery task as an optimization problem over the Segre manifold, the smooth Riemannian manifold of rank-one tensors. This geometric viewpoint yields two powerful algorithms: Riemannian Gradient Descent (RGD) and Riemannian Gauss-Newton (RGN), each of which preserves feasibility at every iteration. Under mild noise assumptions, we prove that RGD converges at a local linear rate, while RGN exhibits an initial local quadratic convergence phase that transitions to a linear rate as the iterates approach the statistical noise floor. Extensive synthetic experiments validate these convergence guarantees and demonstrate the practical effectiveness of our methods.

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


### Learning to Retrieve for Environmental Knowledge Discovery: An Augmentation-Adaptive Self-Supervised Learning Framework
**Date:** 2025-09-19 | **Arxiv:** [2509.14563](https://hub.bitwiki.org/t/learning-to-retrieve-for-environmental-knowledge-discovery-an-augmentation-adaptive-self-supervised-learning-framework/10033)

#### Abstract
The discovery of environmental knowledge depends on labeled task-specific data, but is often constrained by the high cost of data collection. Existing machine learning approaches usually struggle to generalize in data-sparse or atypical conditions. To this end, we propose an Augmentation-Adaptive Self-Supervised Learning (A$^2$SL) framework, which retrieves relevant observational samples to enhance modeling of the target ecosystem. Specifically, we introduce a multi-level pairwise learning loss to train a scenario encoder that captures varying degrees of similarity among scenarios. These learned similarities drive a retrieval mechanism that supplements a target scenario with relevant data from different locations or time periods. Furthermore, to better handle variable scenarios, particularly under atypical or extreme conditions where traditional models struggle, we design an augmentation-adaptive mechanism that selectively enhances these scenarios through targeted data augmentation. Using freshwater ecosystems as a case study, we evaluate A$^2$SL in modeling water temperature and dissolved oxygen dynamics in real-world lakes. Experimental results show that A$^2$SL significantly improves predictive accuracy and enhances robustness in data-scarce and atypical scenarios. Although this study focuses on freshwater ecosystems, the A$^2$SL framework offers a broadly applicable solution in various scientific domains.

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


### Precision Neural Networks: Joint Graph And Relational Learning
**Date:** 2025-09-19 | **Arxiv:** [2509.14821](https://hub.bitwiki.org/t/precision-neural-networks-joint-graph-and-relational-learning/10067)

#### Abstract
CoVariance Neural Networks (VNNs) perform convolutions on the graph determined by the covariance matrix of the data, which enables expressive and stable covariance-based learning. However, covariance matrices are typically dense, fail to encode conditional independence, and are often precomputed in a task-agnostic way, which may hinder performance. To overcome these limitations, we study Precision Neural Networks (PNNs), i.e., VNNs on the precision matrix -- the inverse covariance. The precision matrix naturally encodes statistical independence, often exhibits sparsity, and preserves the covariance spectral structure. To make precision estimation task-aware, we formulate an optimization problem that jointly learns the network parameters and the precision matrix, and solve it via alternating optimization, by sequentially updating the network weights and the precision estimate. We theoretically bound the distance between the estimated and true precision matrices at each iteration, and demonstrate the effectiveness of joint estimation compared to two-step approaches on synthetic and real-world data.

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
* **Limits:** However, covariance matrices are typically dense, fail to encode conditional independence, and are often precomputed in a task-agnostic way, which may hinder performance.
* **Signal Tags:** #ai

---


### Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs
**Date:** 2025-09-16 | **Arxiv:** [2509.10504](https://hub.bitwiki.org/t/retrosynthesis-planning-via-worst-path-policy-optimisation-in-tree-structured-mdps/9385)

#### Abstract
Retrosynthesis planning aims to decompose target molecules into available building blocks, forming a synthetic tree where each internal node represents an intermediate compound and each leaf ideally corresponds to a purchasable reactant. However, this tree becomes invalid if any leaf node is not a valid building block, making the planning process vulnerable to the "weakest link" in the synthetic route. Existing methods often optimise for average performance across branches, failing to account for this worst-case sensitivity. In this paper, we reframe retrosynthesis as a worst-path optimisation problem within tree-structured Markov Decision Processes (MDPs). We prove that this formulation admits a unique optimal solution and provides monotonic improvement guarantees. Building on this insight, we introduce Interactive Retrosynthesis Planning (InterRetro), a method that interacts with the tree MDP, learns a value function for worst-path outcomes, and improves its policy through self-imitation, preferentially reinforcing past decisions with high estimated advantage. Empirically, InterRetro achieves state-of-the-art results - solving 100% of targets on the Retro*-190 benchmark, shortening synthetic routes by 4.9%, and achieving promising performance using only 10% of the training data.

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
* **Limits:** However, this tree becomes invalid if any leaf node is not a valid building block, making the planning process vulnerable to the "weakest link" in the synthetic route.
* **Signal Tags:** #ai

---


### Data-Efficient Time-Dependent PDE Surrogates: Graph Neural Simulators vs. Neural Operators
**Date:** 2025-09-09 | **Arxiv:** [2509.06154](https://hub.bitwiki.org/t/data-efficient-time-dependent-pde-surrogates-graph-neural-simulators-vs-neural-operators/8279)

#### Abstract
Developing accurate, data-efficient surrogate models is central to advancing AI for Science. Neural operators (NOs), which approximate mappings between infinite-dimensional function spaces using conventional neural architectures, have gained popularity as surrogates for systems driven by partial differential equations (PDEs). However, their reliance on large datasets and limited ability to generalize in low-data regimes hinder their practical utility. We argue that these limitations arise from their global processing of data, which fails to exploit the local, discretized structure of physical systems. To address this, we propose Graph Neural Simulators (GNS) as a principled surrogate modeling paradigm for time-dependent PDEs. GNS leverages message-passing combined with numerical time-stepping schemes to learn PDE dynamics by modeling the instantaneous time derivatives. This design mimics traditional numerical solvers, enabling stable long-horizon rollouts and strong inductive biases that enhance generalization. We rigorously evaluate GNS on four canonical PDE systems: (1) 2D scalar Burgers', (2) 2D coupled Burgers', (3) 2D Allen-Cahn, and (4) 2D nonlinear shallow-water equations, comparing against state-of-the-art NOs including Deep Operator Network (DeepONet) and Fourier Neural Operator (FNO). Results demonstrate that GNS is markedly more data-efficient, achieving less than 1% relative L2 error using only 3% of available trajectories, and exhibits dramatically reduced error accumulation over time (82.5% lower autoregressive error than FNO, 99.9% lower than DeepONet). To choose the training data, we introduce a PCA combined with KMeans trajectory selection strategy. These findings provide compelling evidence that GNS, with its graph-based locality and solver-inspired design, is the most suitable and scalable surrogate modeling framework for AI-driven scientific discovery.

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
* **Limits:** However, their reliance on large datasets and limited ability to generalize in low-data regimes hinder their practical utility.
* **Signal Tags:** #ai

---


### Learning to accelerate distributed ADMM using graph neural networks
**Date:** 2025-09-08 | **Arxiv:** [2509.05288](https://hub.bitwiki.org/t/learning-to-accelerate-distributed-admm-using-graph-neural-networks/8097)

#### Abstract
Distributed optimization is fundamental in large-scale machine learning and control applications. Among existing methods, the Alternating Direction Method of Multipliers (ADMM) has gained popularity due to its strong convergence guarantees and suitability for decentralized computation. However, ADMM often suffers from slow convergence and sensitivity to hyperparameter choices. In this work, we show that distributed ADMM iterations can be naturally represented within the message-passing framework of graph neural networks (GNNs). Building on this connection, we propose to learn adaptive step sizes and communication weights by a graph neural network that predicts the hyperparameters based on the iterates. By unrolling ADMM for a fixed number of iterations, we train the network parameters end-to-end to minimize the final iterates error for a given problem class, while preserving the algorithm's convergence properties. Numerical experiments demonstrate that our learned variant consistently improves convergence speed and solution quality compared to standard ADMM. The code is available at https://github.com/paulhausner/learning-distributed-admm.

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
* **Limits:** However, ADMM often suffers from slow convergence and sensitivity to hyperparameter choices.
* **Signal Tags:** #ai

---


### Towards a Unified Textual Graph Framework for Spectral Reasoning via Physical and Chemical Information Fusion
**Date:** 2025-09-03 | **Arxiv:** [2506.17761](https://hub.bitwiki.org/t/towards-a-unified-textual-graph-framework-for-spectral-reasoning-via-physical-and-chemical-information-fusion/7481)

#### Abstract
Motivated by the limitations of current spectral analysis methods-such as reliance on single-modality data, limited generalizability, and poor interpretability-we propose a novel multi-modal spectral analysis framework that integrates prior knowledge graphs with Large Language Models. Our method explicitly bridges physical spectral measurements and chemical structural semantics by representing them in a unified Textual Graph format, enabling flexible, interpretable, and generalizable spectral understanding. Raw spectra are first transformed into TAGs, where nodes and edges are enriched with textual attributes describing both spectral properties and chemical context. These are then merged with relevant prior knowledge-including functional groups and molecular graphs-to form a Task Graph that incorporates "Prompt Nodes" supporting LLM-based contextual reasoning. A Graph Neural Network further processes this structure to complete downstream tasks. This unified design enables seamless multi-modal integration and automated feature decoding with minimal manual annotation. Our framework achieves consistently high performance across multiple spectral analysis tasks, including node-level, edge-level, and graph-level classification. It demonstrates robust generalization in both zero-shot and few-shot settings, highlighting its effectiveness in learning from limited data and supporting in-context reasoning. This work establishes a scalable and interpretable foundation for LLM-driven spectral analysis, unifying physical and chemical modalities for scientific applications.

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
