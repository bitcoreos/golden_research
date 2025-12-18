# Vol 17 Graph   Topology
*Enriched by BITCOREOS | Phase 4 Batch 4*

---

### Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings
**Date:** 2025-12-08 | **Arxiv:** [2505.18973](https://arxiv.org/abs/2505.18973)

#### Abstract
Selective state-space models excel at long-sequence modeling, but their capacity for language representation -- in complex hierarchical reasoning -- remains underexplored. Most large language models rely on \textit{flat} Euclidean embeddings, limiting their ability to capture latent hierarchies. To address this, we propose {\it Hierarchical Mamba (HiM)}, integrating efficient Mamba2 with hyperbolic geometry to learn hierarchy-aware language embeddings for deeper linguistic understanding. Mamba2-processed sequences are projected to the Poincaré ball or Lorentzian manifold with ``learnable'' curvature, optimized with a hyperbolic loss. Our HiM model facilitates the capture of relational distances across varying hierarchical levels, enabling effective long-range reasoning for tasks like mixed-hop prediction and multi-hop inference in hierarchical classification. Experimental results show both HiM variants effectively capture hierarchical relationships across four linguistic and medical datasets, surpassing Euclidean baselines, with HiM-Poincaré providing fine-grained distinctions with higher h-norms, while HiM-Lorentz offers more stable, compact, and hierarchy-preserving embeddings-favoring robustness. The source code is publicly available at https://github.com/BerryByte/HiM.

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


### Don't Reach for the Stars: Rethinking Topology for Resilient Federated Learning
**Date:** 2025-11-25 | **Arxiv:** [2508.05224](https://arxiv.org/abs/2508.05224)

#### Abstract
Federated learning (FL) enables collaborative model training across distributed clients while preserving data privacy by keeping data local. Traditional FL approaches rely on a centralized, star-shaped topology, where a central server aggregates model updates from clients. However, this architecture introduces several limitations, including a single point of failure, limited personalization, and poor robustness to distribution shifts or vulnerability to malfunctioning clients. Moreover, update selection in centralized FL often relies on low-level parameter differences, which can be unreliable when client data is not independent and identically distributed, and offer clients little control. In this work, we propose a decentralized, peer-to-peer (P2P) FL framework. It leverages the flexibility of the P2P topology to enable each client to identify and aggregate a personalized set of trustworthy and beneficial updates.This framework is the Local Inference Guided Aggregation for Heterogeneous Training Environments to Yield Enhancement Through Agreement and Regularization (LIGHTYEAR). Central to our method is an agreement score, computed on a local validation set, which quantifies the semantic alignment of incoming updates in the function space with respect to the clients reference model. Each client uses this score to select a tailored subset of updates and performs aggregation with a regularization term that further stabilizes the training. Our empirical evaluation across five datasets shows that the proposed approach consistently outperforms both, centralized baselines and existing P2P methods in terms of client-level performance, particularly under adversarial and heterogeneous conditions.

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
* **Limits:** However, this architecture introduces several limitations, including a single point of failure, limited personalization, and poor robustness to distribution shifts or vulnerability to malfunctioning clients.
* **Signal Tags:** #ai

---


### Topology-Aware Active Learning on Graphs
**Date:** 2025-10-31 | **Arxiv:** [2510.25892](https://arxiv.org/abs/2510.25892)

#### Abstract
We propose a graph-topological approach to active learning that directly targets the core challenge of exploration versus exploitation under scarce label budgets. To guide exploration, we introduce a coreset construction algorithm based on Balanced Forman Curvature (BFC), which selects representative initial labels that reflect the graph's cluster structure. This method includes a data-driven stopping criterion that signals when the graph has been sufficiently explored. We further use BFC to dynamically trigger the shift from exploration to exploitation within active learning routines, replacing hand-tuned heuristics. To improve exploitation, we introduce a localized graph rewiring strategy that efficiently incorporates multiscale information around labeled nodes, enhancing label propagation while preserving sparsity. Experiments on benchmark classification tasks show that our methods consistently outperform existing graph-based semi-supervised baselines at low label rates.

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


### Constructive Lyapunov Functions via Topology-Preserving Neural Networks
**Date:** 2025-10-30 | **Arxiv:** [2510.24730](https://arxiv.org/abs/2510.24730)

#### Abstract
We prove that ONN achieves order-optimal performance on convergence rate ($μ\propto λ_2$), edge efficiency ($E = N$ for minimal connectivity $k = 2$), and computational complexity ($O(N d^2)$). Empirical validation on 3M-node semantic networks demonstrates 99.75\% improvement over baseline methods, confirming exponential convergence ($μ= 3.2 \times 10^{-4}$) and topology preservation. ORTSF integration into transformers achieves 14.7\% perplexity reduction and 2.3 faster convergence on WikiText-103. We establish deep connections to optimal control (Hamilton-Jacobi-Bellman), information geometry (Fisher-efficient natural gradient), topological data analysis (persistent homology computation in $O(KN)$), discrete geometry (Ricci flow), and category theory (adjoint functors). This work transforms Massera's abstract existence theorem into a concrete, scalable algorithm with provable guarantees, opening pathways for constructive stability analysis in neural networks, robotics, and distributed systems.

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
* **Construct:** Embodied AI
* **Layer:** Theory
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Hyperbolic Structured Classification for Robust Single Positive Multi-label Learning
**Date:** 2025-10-20 | **Arxiv:** [2510.15296](https://arxiv.org/abs/2510.15296)

#### Abstract
Single Positive Multi-Label Learning (SPMLL) addresses the challenging scenario where each training sample is annotated with only one positive label despite potentially belonging to multiple categories, making it difficult to capture complex label relationships and hierarchical structures. While existing methods implicitly model label relationships through distance-based similarity, lacking explicit geometric definitions for different relationship types. To address these limitations, we propose the first hyperbolic classification framework for SPMLL that represents each label as a hyperbolic ball rather than a point or vector, enabling rich inter-label relationship modeling through geometric ball interactions. Our ball-based approach naturally captures multiple relationship types simultaneously: inclusion for hierarchical structures, overlap for co-occurrence patterns, and separation for semantic independence. Further, we introduce two key component innovations: a temperature-adaptive hyperbolic ball classifier and a physics-inspired double-well regularization that guides balls toward meaningful configurations. To validate our approach, extensive experiments on four benchmark datasets (MS-COCO, PASCAL VOC, NUS-WIDE, CUB-200-2011) demonstrate competitive performance with superior interpretability compared to existing methods. Furthermore, statistical analysis reveals strong correlation between learned embeddings and real-world co-occurrence patterns, establishing hyperbolic geometry as a more robust paradigm for structured classification under incomplete supervision.

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
* **Construct:** Machine Perception
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Graph Neural Networks for Transmission Grid Topology Control: Busbar Information Asymmetry and Heterogeneous Representations
**Date:** 2025-10-06 | **Arxiv:** [2501.07186](https://arxiv.org/abs/2501.07186)

#### Abstract
Factors such as the proliferation of renewable energy and electrification contribute to grid congestion as a pressing problem. Topology control is an appealing method for relieving congestion, but traditional approaches for topology discovery have proven too slow for practical application. Recent research has focused on machine learning (ML) as an efficient alternative. Graph neural networks (GNNs) are particularly well-suited for topology control applications due to their ability to model the graph structure of power grids. This study investigates the effect of the graph representation on GNN effectiveness for topology control. We identify the busbar information asymmetry problem inherent to the popular homogeneous graph representation. We propose a heterogeneous graph representation that resolves this problem. We apply GNNs with both representations and a fully connected neural network (FCNN) baseline on an imitation learning task. The models are evaluated by classification accuracy and grid operation ability. We find that heterogeneous GNNs perform best on in-distribution network configurations, followed by FCNNs, and lastly, homogeneous GNNs. We also find that both GNN types generalize better to out-of-distribution network configurations than FCNNs.

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


### Safeguarding Graph Neural Networks against Topology Inference Attacks
**Date:** 2025-09-09 | **Arxiv:** [2509.05429](https://arxiv.org/abs/2509.05429)

#### Abstract
Graph Neural Networks (GNNs) have emerged as powerful models for learning from graph-structured data. However, their widespread adoption has raised serious privacy concerns. While prior research has primarily focused on edge-level privacy, a critical yet underexplored threat lies in topology privacy - the confidentiality of the graph's overall structure. In this work, we present a comprehensive study on topology privacy risks in GNNs, revealing their vulnerability to graph-level inference attacks. To this end, we propose a suite of Topology Inference Attacks (TIAs) that can reconstruct the structure of a target training graph using only black-box access to a GNN model. Our findings show that GNNs are highly susceptible to these attacks, and that existing edge-level differential privacy mechanisms are insufficient as they either fail to mitigate the risk or severely compromise model accuracy. To address this challenge, we introduce Private Graph Reconstruction (PGR), a novel defense framework designed to protect topology privacy while maintaining model accuracy. PGR is formulated as a bi-level optimization problem, where a synthetic training graph is iteratively generated using meta-gradients, and the GNN model is concurrently updated based on the evolving graph. Extensive experiments demonstrate that PGR significantly reduces topology leakage with minimal impact on model accuracy. Our code is available at https://github.com/JeffffffFu/PGR.

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
* **Limits:** However, their widespread adoption has raised serious privacy concerns.
* **Signal Tags:** #ai

---


### Towards Heterogeneity-Aware and Energy-Efficient Topology Optimization for Decentralized Federated Learning in Edge Environment
**Date:** 2025-08-13 | **Arxiv:** [2508.08278](https://arxiv.org/abs/2508.08278)

#### Abstract
Federated learning (FL) has emerged as a promising paradigm within edge computing (EC) systems, enabling numerous edge devices to collaboratively train artificial intelligence (AI) models while maintaining data privacy. To overcome the communication bottlenecks associated with centralized parameter servers, decentralized federated learning (DFL), which leverages peer-to-peer (P2P) communication, has been extensively explored in the research community. Although researchers design a variety of DFL approach to ensure model convergence, its iterative learning process inevitably incurs considerable cost along with the growth of model complexity and the number of participants. These costs are largely influenced by the dynamic changes of topology in each training round, particularly its sparsity and connectivity conditions. Furthermore, the inherent resources heterogeneity in the edge environments affects energy efficiency of learning process, while data heterogeneity degrades model performance. These factors pose significant challenges to the design of an effective DFL framework for EC systems. To this end, we propose Hat-DFed, a heterogeneity-aware and coset-effective decentralized federated learning (DFL) framework. In Hat-DFed, the topology construction is formulated as a dual optimization problem, which is then proven to be NP-hard, with the goal of maximizing model performance while minimizing cumulative energy consumption in complex edge environments. To solve this problem, we design a two-phase algorithm that dynamically constructs optimal communication topologies while unbiasedly estimating their impact on both model performance and energy cost. Additionally, the algorithm incorporates an importance-aware model aggregation mechanism to mitigate performance degradation caused by data heterogeneity.

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


### RGE-GCN: Recursive Gene Elimination with Graph Convolutional Networks for RNA-seq based Early Cancer Detection
**Date:** 2025-12-05 | **Arxiv:** [2512.04333](https://arxiv.org/abs/2512.04333)

#### Abstract
Early detection of cancer plays a key role in improving survival rates, but identifying reliable biomarkers from RNA-seq data is still a major challenge. The data are high-dimensional, and conventional statistical methods often fail to capture the complex relationships between genes. In this study, we introduce RGE-GCN (Recursive Gene Elimination with Graph Convolutional Networks), a framework that combines feature selection and classification in a single pipeline. Our approach builds a graph from gene expression profiles, uses a Graph Convolutional Network to classify cancer versus normal samples, and applies Integrated Gradients to highlight the most informative genes. By recursively removing less relevant genes, the model converges to a compact set of biomarkers that are both interpretable and predictive. We evaluated RGE-GCN on synthetic data as well as real-world RNA-seq cohorts of lung, kidney, and cervical cancers. Across all datasets, the method consistently achieved higher accuracy and F1-scores than standard tools such as DESeq2, edgeR, and limma-voom. Importantly, the selected genes aligned with well-known cancer pathways including PI3K-AKT, MAPK, SUMOylation, and immune regulation. These results suggest that RGE-GCN shows promise as a generalizable approach for RNA-seq based early cancer detection and biomarker discovery (https://rce-gcn.streamlit.app/ ).

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


### FAST: Topology-Aware Frequency-Domain Distribution Matching for Coreset Selection
**Date:** 2025-11-26 | **Arxiv:** [2511.19476](https://arxiv.org/abs/2511.19476)

#### Abstract
Coreset selection compresses large datasets into compact, representative subsets, reducing the energy and computational burden of training deep neural networks. Existing methods are either: (i) DNN-based, which are tied to model-specific parameters and introduce architectural bias; or (ii) DNN-free, which rely on heuristics lacking theoretical guarantees. Neither approach explicitly constrains distributional equivalence, largely because continuous distribution matching is considered inapplicable to discrete sampling. Moreover, prevalent metrics (e.g., MSE, KL, MMD, CE) cannot accurately capture higher-order moment discrepancies, leading to suboptimal coresets. In this work, we propose FAST, the first DNN-free distribution-matching coreset selection framework that formulates the coreset selection task as a graph-constrained optimization problem grounded in spectral graph theory and employs the Characteristic Function Distance (CFD) to capture full distributional information in the frequency domain. We further discover that naive CFD suffers from a "vanishing phase gradient" issue in medium and high-frequency regions; to address this, we introduce an Attenuated Phase-Decoupled CFD. Furthermore, for better convergence, we design a Progressive Discrepancy-Aware Sampling strategy that progressively schedules frequency selection from low to high, preserving global structure before refining local details and enabling accurate matching with fewer frequencies while avoiding overfitting. Extensive experiments demonstrate that FAST significantly outperforms state-of-the-art coreset selection methods across all evaluated benchmarks, achieving an average accuracy gain of 9.12%. Compared to other baseline coreset methods, it reduces power consumption by 96.57% and achieves a 2.2x average speedup, underscoring its high performance and energy efficiency.

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


### IIKL: Isometric Immersion Kernel Learning with Riemannian Manifold for Geometric Preservation
**Date:** 2025-11-25 | **Arxiv:** [2505.06288](https://arxiv.org/abs/2505.06288)

#### Abstract
Geometric representation learning in preserving the intrinsic geometric and topological properties for discrete non-Euclidean data is crucial in scientific applications. Previous research generally mapped non-Euclidean discrete data into Euclidean space during representation learning, which may lead to the loss of some critical geometric information. In this paper, we propose a novel Isometric Immersion Kernel Learning (IIKL) method to build Riemannian manifold and isometrically induce Riemannian metric from discrete non-Euclidean data. We prove that Isometric immersion is equivalent to the kernel function in the tangent bundle on the manifold, which explicitly guarantees the invariance of the inner product between vectors in the arbitrary tangent space throughout the learning process, thus maintaining the geometric structure of the original data. Moreover, a novel parameterized learning model based on IIKL is introduced, and an alternating training method for this model is derived using Maximum Likelihood Estimation (MLE), ensuring efficient convergence. Experimental results proved that using the learned Riemannian manifold and its metric, our model preserved the intrinsic geometric representation of data in both 3D and high-dimensional datasets successfully, and significantly improved the accuracy of downstream tasks, such as data reconstruction and classification. It is showed that our method could reduce the inner product invariant loss by more than 90% compared to state-of-the-art (SOTA) methods, also achieved an average 40% improvement in downstream reconstruction accuracy and a 90% reduction in error for geometric metrics involving isometric and conformal.

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


### Uncertainty of Network Topology with Applications to Out-of-Distribution Detection
**Date:** 2025-11-25 | **Arxiv:** [2511.18813](https://arxiv.org/abs/2511.18813)

#### Abstract
Persistent homology (PH) is a crucial concept in computational topology, providing a multiscale topological description of a space. It is particularly significant in topological data analysis, which aims to make statistical inference from a topological perspective. In this work, we introduce a new topological summary for Bayesian neural networks, termed the predictive topological uncertainty (pTU). The proposed pTU measures the uncertainty in the interaction between the model and the inputs. It provides insights from the model perspective: if two samples interact with a model in a similar way, then they are considered identically distributed. We also show that the pTU is insensitive to the model architecture. As an application, pTU is used to solve the out-of-distribution (OOD) detection problem, which is critical to ensure model reliability. Failure to detect OOD input can lead to incorrect and unreliable predictions. To address this issue, we propose a significance test for OOD based on the pTU, providing a statistical framework for this issue. The effectiveness of the framework is validated through various experiments, in terms of its statistical power, sensitivity, and robustness.

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


### Integrating Causal Inference with Graph Neural Networks for Alzheimer's Disease Analysis
**Date:** 2025-11-20 | **Arxiv:** [2511.14922](https://arxiv.org/abs/2511.14922)

#### Abstract
Deep graph learning has advanced Alzheimer's (AD) disease classification from MRI, but most models remain correlational, confounding demographic and genetic factors with disease specific features. We present Causal-GCN, an interventional graph convolutional framework that integrates do-calculus-based back-door adjustment to identify brain regions exerting stable causal influence on AD progression. Each subject's MRI is represented as a structural connectome where nodes denote cortical and subcortical regions and edges encode anatomical connectivity. Confounders such as age, sec, and APOE4 genotype are summarized via principal components and included in the causal adjustment set. After training, interventions on individual regions are simulated by serving their incoming edges and altering node features to estimate average causal effects on disease probability. Applied to 484 subjects from the ADNI cohort, Causal-GCN achieves performance comparable to baseline GNNs while providing interpretable causal effect rankings that highlight posterior, cingulate, and insular hubs consistent with established AD neuropathology.

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


### Causal Inference, Biomarker Discovery, Graph Neural Network, Feature Selection
**Date:** 2025-11-18 | **Arxiv:** [2511.13295](https://arxiv.org/abs/2511.13295)

#### Abstract
Biomarker discovery from high-throughput transcriptomic data is crucial for advancing precision medicine. However, existing methods often neglect gene-gene regulatory relationships and lack stability across datasets, leading to conflation of spurious correlations with genuine causal effects. To address these issues, we develop a causal graph neural network (Causal-GNN) method that integrates causal inference with multi-layer graph neural networks (GNNs). The key innovation is the incorporation of causal effect estimation for identifying stable biomarkers, coupled with a GNN-based propensity scoring mechanism that leverages cross-gene regulatory networks. Experimental results demonstrate that our method achieves consistently high predictive accuracy across four distinct datasets and four independent classifiers. Moreover, it enables the identification of more stable biomarkers compared to traditional methods. Our work provides a robust, efficient, and biologically interpretable tool for biomarker discovery, demonstrating strong potential for broad application across medical disciplines.

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
* **Limits:** However, existing methods often neglect gene-gene regulatory relationships and lack stability across datasets, leading to conflation of spurious correlations with genuine causal effects.
* **Signal Tags:** #ai

---


### Topology-Aware Conformal Prediction for Stream Networks
**Date:** 2025-11-11 | **Arxiv:** [2503.04981](https://arxiv.org/abs/2503.04981)

#### Abstract
Stream networks, a unique class of spatiotemporal graphs, exhibit complex directional flow constraints and evolving dependencies, making uncertainty quantification a critical yet challenging task. Traditional conformal prediction methods struggle in this setting due to the need for joint predictions across multiple interdependent locations and the intricate spatio-temporal dependencies inherent in stream networks. Existing approaches either neglect dependencies, leading to overly conservative predictions, or rely solely on data-driven estimations, failing to capture the rich topological structure of the network. To address these challenges, we propose Spatio-Temporal Adaptive Conformal Inference (\texttt{STACI}), a novel framework that integrates network topology and temporal dynamics into the conformal prediction framework. \texttt{STACI} introduces a topology-aware nonconformity score that respects directional flow constraints and dynamically adjusts prediction sets to account for temporal distributional shifts. We provide theoretical guarantees on the validity of our approach and demonstrate its superior performance on both synthetic and real-world datasets. Our results show that \texttt{STACI} effectively balances prediction efficiency and coverage, outperforming existing conformal prediction methods for stream networks.

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


### Causal Graph Neural Networks for Healthcare
**Date:** 2025-11-05 | **Arxiv:** [2511.02531](https://arxiv.org/abs/2511.02531)

#### Abstract
Healthcare artificial intelligence systems routinely fail when deployed across institutions, with documented performance drops and perpetuation of discriminatory patterns embedded in historical data. This brittleness stems, in part, from learning statistical associations rather than causal mechanisms. Causal graph neural networks address this triple crisis of distribution shift, discrimination, and inscrutability by combining graph-based representations of biomedical data with causal inference principles to learn invariant mechanisms rather than spurious correlations. This Review examines methodological foundations spanning structural causal models, disentangled causal representation learning, and techniques for interventional prediction and counterfactual reasoning on graphs. We analyse applications demonstrating clinical value across psychiatric diagnosis through brain network analysis, cancer subtyping via multi-omics causal integration, continuous physiological monitoring with mechanistic interpretation, and drug recommendation correcting prescription bias. These advances establish foundations for patient-specific Causal Digital Twins, enabling in silico clinical experimentation, with integration of large language models for hypothesis generation and causal graph neural networks for mechanistic validation. Substantial barriers remain, including computational requirements precluding real-time deployment, validation challenges demanding multi-modal evidence triangulation beyond cross-validation, and risks of causal-washing where methods employ causal terminology without rigorous evidentiary support. We propose tiered frameworks distinguishing causally-inspired architectures from causally-validated discoveries and identify critical research priorities making causal rather than purely associational claims.

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


### The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute
**Date:** 2025-11-05 | **Arxiv:** [2511.02309](https://arxiv.org/abs/2511.02309)

#### Abstract
We revisit test-time scaling for language model reasoning and ask a fundamental question: at equal token budget and compute, is it better to run multiple independent chains in parallel, or to run fewer chains that iteratively refine through sequential steps? Through comprehensive evaluation across 5 state-of-the-art open source models and 3 challenging reasoning benchmarks, we find that sequential scaling where chains explicitly build upon previous attempts consistently outperforms the dominant parallel self-consistency paradigm in 95.6% of configurations with gains in accuracy upto 46.7%. Further, we introduce inverse-entropy weighted voting, a novel training-free method to further boost the accuracy of sequential scaling. By weighing answers in proportion to the inverse entropy of their reasoning chains, we increase our success rate over parallel majority and establish it as the optimal test-time scaling strategy. Our findings fundamentally challenge the parallel reasoning orthodoxy that has dominated test-time scaling since Wang et al.'s self-consistency decoding (Wang et al., 2022), positioning sequential refinement as the robust default for modern LLM reasoning and necessitating a paradigm shift in how we approach inference-time optimization.

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


### On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix Multiplication
**Date:** 2025-11-04 | **Arxiv:** [2511.00025](https://arxiv.org/abs/2511.00025)

#### Abstract
Floating-point non-associativity makes fundamental deep learning operations, such as matrix multiplication (matmul) on GPUs, inherently non-deterministic. Despite this, the statistical structure of the resulting numerical error remains poorly understood. A common working assumption is that these errors behave as independent and identically distributed (i.i.d.) Gaussian noise. In this paper, we empirically test this assumption and show that it fails to describe real GPU behavior. By comparing outputs of single-input and batched matmuls, we find that while the i.i.d. model predicts non-zero output instability, empirical results show a 0.00% prediction flip rate. Through covariance analysis, we uncover the cause: the floating-point error is structured and highly correlated. For float16, nearly 50% of the total error variance lies in off-diagonal terms, revealing that the noise behaves as a coordinated, directional perturbation rather than random static. This result challenges the prevailing stochastic view of numerical noise and provides a principled foundation for analyzing deep learning reliability under hardware non-determinism.

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


### MaGNet: A Mamba Dual-Hypergraph Network for Stock Prediction via Temporal-Causal and Global Relational Learning
**Date:** 2025-11-04 | **Arxiv:** [2511.00085](https://arxiv.org/abs/2511.00085)

#### Abstract
Stock trend prediction is crucial for profitable trading strategies and portfolio management yet remains challenging due to market volatility, complex temporal dynamics and multifaceted inter-stock relationships. Existing methods struggle to effectively capture temporal dependencies and dynamic inter-stock interactions, often neglecting cross-sectional market influences, relying on static correlations, employing uniform treatments of nodes and edges, and conflating diverse relationships. This work introduces MaGNet, a novel Mamba dual-hyperGraph Network for stock prediction, integrating three key innovations: (1) a MAGE block, which leverages bidirectional Mamba with adaptive gating mechanisms for contextual temporal modeling and integrates a sparse Mixture-of-Experts layer to enable dynamic adaptation to diverse market conditions, alongside multi-head attention for capturing global dependencies; (2) Feature-wise and Stock-wise 2D Spatiotemporal Attention modules enable precise fusion of multivariate features and cross-stock dependencies, effectively enhancing informativeness while preserving intrinsic data structures, bridging temporal modeling with relational reasoning; and (3) a dual hypergraph framework consisting of the Temporal-Causal Hypergraph (TCH) that captures fine-grained causal dependencies with temporal constraints, and Global Probabilistic Hypergraph (GPH) that models market-wide patterns through soft hyperedge assignments and Jensen-Shannon Divergence weighting mechanism, jointly disentangling localized temporal influences from instantaneous global structures for multi-scale relational learning. Extensive experiments on six major stock indices demonstrate MaGNet outperforms state-of-the-art methods in both superior predictive performance and exceptional investment returns with robust risk management capabilities. Codes available at: https://github.com/PeilinTime/MaGNet.

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


### Solution Space Topology Guides CMTS Search
**Date:** 2025-11-04 | **Arxiv:** [2511.01701](https://arxiv.org/abs/2511.01701)

#### Abstract
A fundamental question in search-guided AI: what topology should guide Monte Carlo Tree Search (MCTS) in puzzle solving? Prior work applied topological features to guide MCTS in ARC-style tasks using grid topology -- the Laplacian spectral properties of cell connectivity -- and found no benefit. We identify the root cause: grid topology is constant across all instances. We propose measuring \emph{solution space topology} instead: the structure of valid color assignments constrained by detected pattern rules. We build this via compatibility graphs where nodes are $(cell, color)$ pairs and edges represent compatible assignments under pattern constraints.   Our method: (1) detect pattern rules automatically with 100\% accuracy on 5 types, (2) construct compatibility graphs encoding solution space structure, (3) extract topological features (algebraic connectivity, rigidity, color structure) that vary with task difficulty, (4) integrate these features into MCTS node selection via sibling-normalized scores.   We provide formal definitions, a rigorous selection formula, and comprehensive ablations showing that algebraic connectivity is the dominant signal. The work demonstrates that topology matters for search -- but only the \emph{right} topology. For puzzle solving, this is solution space structure, not problem space structure.

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


### Taming the Tail: NoI Topology Synthesis for Mixed DL Workloads on Chiplet-Based Accelerators
**Date:** 2025-10-29 | **Arxiv:** [2510.24113](https://arxiv.org/abs/2510.24113)

#### Abstract
Heterogeneous chiplet-based systems improve scaling by disag-gregating CPUs/GPUs and emerging technologies (HBM/DRAM).However this on-package disaggregation introduces a latency inNetwork-on-Interposer(NoI). We observe that in modern large-modelinference, parameters and activations routinely move backand forth from HBM/DRAM, injecting large, bursty flows into theinterposer. These memory-driven transfers inflate tail latency andviolate Service Level Agreements (SLAs) across k-ary n-cube base-line NoI topologies. To address this gap we introduce an InterferenceScore (IS) that quantifies worst-case slowdown under contention.We then formulate NoI synthesis as a multi-objective optimization(MOO) problem. We develop PARL (Partition-Aware ReinforcementLearner), a topology generator that balances throughput, latency,and power. PARL-generated topologies reduce contention at the memory cut, meet SLAs, and cut worst-case slowdown to 1.2 times while maintaining competitive mean throughput relative to link-rich meshes. Overall, this reframes NoI design for heterogeneouschiplet accelerators with workload-aware objectives.

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


### TopoFR: A Closer Look at Topology Alignment on Face Recognition
**Date:** 2025-10-27 | **Arxiv:** [2410.10587](https://arxiv.org/abs/2410.10587)

#### Abstract
The field of face recognition (FR) has undergone significant advancements with the rise of deep learning. Recently, the success of unsupervised learning and graph neural networks has demonstrated the effectiveness of data structure information. Considering that the FR task can leverage large-scale training data, which intrinsically contains significant structure information, we aim to investigate how to encode such critical structure information into the latent space. As revealed from our observations, directly aligning the structure information between the input and latent spaces inevitably suffers from an overfitting problem, leading to a structure collapse phenomenon in the latent space. To address this problem, we propose TopoFR, a novel FR model that leverages a topological structure alignment strategy called PTSA and a hard sample mining strategy named SDE. Concretely, PTSA uses persistent homology to align the topological structures of the input and latent spaces, effectively preserving the structure information and improving the generalization performance of FR model. To mitigate the impact of hard samples on the latent space structure, SDE accurately identifies hard samples by automatically computing structure damage score (SDS) for each sample, and directs the model to prioritize optimizing these samples. Experimental results on popular face benchmarks demonstrate the superiority of our TopoFR over the state-of-the-art methods. Code and models are available at: https://github.com/modelscope/facechain/tree/main/face_module/TopoFR.

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


### Topology of Currencies: Persistent Homology for FX Co-movements: A Comparative Clustering Study
**Date:** 2025-10-23 | **Arxiv:** [2510.19306](https://arxiv.org/abs/2510.19306)

#### Abstract
This study investigates whether Topological Data Analysis (TDA) can provide additional insights beyond traditional statistical methods in clustering currency behaviours. We focus on the foreign exchange (FX) market, which is a complex system often exhibiting non-linear and high-dimensional dynamics that classical techniques may not fully capture. We compare clustering results based on TDA-derived features versus classical statistical features using monthly logarithmic returns of 13 major currency exchange rates (all against the euro). Two widely-used clustering algorithms, \(k\)-means and Hierarchical clustering, are applied on both types of features, and cluster quality is evaluated via the Silhouette score and the Calinski-Harabasz index. Our findings show that TDA-based feature clustering produces more compact and well-separated clusters than clustering on traditional statistical features, particularly achieving substantially higher Calinski-Harabasz scores. However, all clustering approaches yield modest Silhouette scores, underscoring the inherent difficulty of grouping FX time series. The differing cluster compositions under TDA vs. classical features suggest that TDA captures structural patterns in currency co-movements that conventional methods might overlook. These results highlight TDA as a valuable complementary tool for analysing financial time series, with potential applications in risk management where understanding structural co-movements is crucial.

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
* **Limits:** However, all clustering approaches yield modest Silhouette scores, underscoring the inherent difficulty of grouping FX time series.
* **Signal Tags:** #ai

---


### Graph Learning is Suboptimal in Causal Bandits
**Date:** 2025-10-21 | **Arxiv:** [2510.16811](https://arxiv.org/abs/2510.16811)

#### Abstract
We study regret minimization in causal bandits under causal sufficiency where the underlying causal structure is not known to the agent. Previous work has focused on identifying the reward's parents and then applying classic bandit methods to them, or jointly learning the parents while minimizing regret. We investigate whether such strategies are optimal. Somewhat counterintuitively, our results show that learning the parent set is suboptimal. We do so by proving that there exist instances where regret minimization and parent identification are fundamentally conflicting objectives. We further analyze both the known and unknown parent set size regimes, establish novel regret lower bounds that capture the combinatorial structure of the action space. Building on these insights, we propose nearly optimal algorithms that bypass graph and parent recovery, demonstrating that parent identification is indeed unnecessary for regret minimization. Experiments confirm that there exists a large performance gap between our method and existing baselines in various environments.

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


### Intrinsic Dimensionality of Fermi-Pasta-Ulam-Tsingou High-Dimensional Trajectories Through Manifold Learning: A Linear Approach
**Date:** 2025-10-21 | **Arxiv:** [2411.02058](https://arxiv.org/abs/2411.02058)

#### Abstract
A data-driven approach based on unsupervised machine learning is proposed to infer the intrinsic dimension $m^{\ast}$ of the high-dimensional trajectories of the Fermi-Pasta-Ulam-Tsingou (FPUT) model. Principal component analysis (PCA) is applied to trajectory data consisting of $n_s = 4,000,000$ datapoints, of the FPUT $β$ model with $N = 32$ coupled oscillators, revealing a critical relationship between $m^{\ast}$ and the model's nonlinear strength. By estimating the intrinsic dimension $m^{\ast}$ using multiple methods (participation ratio, Kaiser rule, and the Kneedle algorithm), it is found that $m^{\ast}$ increases with the model nonlinearity. Interestingly, in the weakly nonlinear regime, for trajectories initialized by exciting the first mode, the participation ratio estimates $m^{\ast} = 2, 3$, strongly suggesting that quasi-periodic motion on a low-dimensional Riemannian manifold underlies the characteristic energy recurrences observed in the FPUT model.

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


### Atlas-based Manifold Representations for Interpretable Riemannian Machine Learning
**Date:** 2025-10-21 | **Arxiv:** [2510.17772](https://arxiv.org/abs/2510.17772)

#### Abstract
Despite the popularity of the manifold hypothesis, current manifold-learning methods do not support machine learning directly on the latent $d$-dimensional data manifold, as they primarily aim to perform dimensionality reduction into $\mathbb{R}^D$, losing key manifold features when the embedding dimension $D$ approaches $d$.   On the other hand, methods that directly learn the latent manifold as a differentiable atlas have been relatively underexplored.   In this paper, we aim to give a proof of concept of the effectiveness and potential of atlas-based methods. To this end, we implement a generic data structure to maintain a differentiable atlas that enables Riemannian optimization over the manifold. We complement this with an unsupervised heuristic that learns a differentiable atlas from point cloud data. We experimentally demonstrate that this approach has advantages in terms of efficiency and accuracy in selected settings. Moreover, in a supervised classification task over the Klein bottle and in RNA velocity analysis of hematopoietic data, we showcase the improved interpretability and robustness of our approach.

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


### TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning
**Date:** 2025-10-15 | **Arxiv:** [2505.15692](https://arxiv.org/abs/2505.15692)

#### Abstract
Reinforcement learning (RL) has emerged as an effective paradigm for enhancing model reasoning. However, existing RL methods like GRPO often rely on unstructured self-sampling to fit scalar rewards, often producing inefficient rollouts that fail to capture transferable problem-solving strategies. To address these limitations, we propose **TemplateRL**, a structured template-guided RL framework that augments policy optimization with explicit template guidance. Our approach first constructs a problem-solving template library via MCTS on a small seed set, then seamlessly integrates this high-level structured guidance into RL training. By guiding rollout generation to align with proven template structures, TemplateRL significantly improves high-quality trajectory hit rates while reducing ineffective exploration. This structure-guided design steers the policy toward validated strategic patterns, stabilizing training dynamics, and enhancing RL sampling efficiency. Notably, the explicit template library is interpretable, editable, and supports online updates-enabling continuous updates during both training and inference. Extensive experiments demonstrate that TemplateRL outperforms GRPO by 99% on AIME and 41% on AMC, with superior stability on weak models and remarkable cross-domain generalization, highlighting its potential for broader tasks.

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
* **Limits:** However, existing RL methods like GRPO often rely on unstructured self-sampling to fit scalar rewards, often producing inefficient rollouts that fail to capture transferable problem-solving strategies.
* **Signal Tags:** #ai

---


### Supervised Manifold Learning for Functional Data
**Date:** 2025-10-14 | **Arxiv:** [2503.17943](https://arxiv.org/abs/2503.17943)

#### Abstract
Classification is a core topic in functional data analysis. A large number of functional classifiers have been proposed in the literature, most of which are based on functional principal component analysis or functional regression. In contrast, we investigate this topic from the perspective of manifold learning. It is assumed that functional data lie on an unknown low-dimensional manifold, and we expect that superior classifiers can be developed based on the manifold structure. To this end, we propose a novel proximity measure that takes the label information into account to learn the low-dimensional representations, also known as the supervised manifold learning outcomes. When the outcomes are coupled with multivariate classifiers, the procedure induces a new family of functional classifiers. In theory, we prove that our functional classifier induced by the $k$-NN classifier is asymptotically optimal. In practice, we show that our method, coupled with several classical multivariate classifiers, achieves highly competitive classification performance compared to existing functional classifiers across both synthetic and real data examples. Supplementary materials are available online.

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


### Taxonomy-aware Dynamic Motion Generation on Hyperbolic Manifolds
**Date:** 2025-09-26 | **Arxiv:** [2509.21281](https://arxiv.org/abs/2509.21281)

#### Abstract
Human-like motion generation for robots often draws inspiration from biomechanical studies, which often categorize complex human motions into hierarchical taxonomies. While these taxonomies provide rich structural information about how movements relate to one another, this information is frequently overlooked in motion generation models, leading to a disconnect between the generated motions and their underlying hierarchical structure. This paper introduces the \ac{gphdm}, a novel approach that learns latent representations preserving both the hierarchical structure of motions and their temporal dynamics to ensure physical consistency. Our model achieves this by extending the dynamics prior of the Gaussian Process Dynamical Model (GPDM) to the hyperbolic manifold and integrating it with taxonomy-aware inductive biases. Building on this geometry- and taxonomy-aware frameworks, we propose three novel mechanisms for generating motions that are both taxonomically-structured and physically-consistent: two probabilistic recursive approaches and a method based on pullback-metric geodesics. Experiments on generating realistic motion sequences on the hand grasping taxonomy show that the proposed GPHDM faithfully encodes the underlying taxonomy and temporal dynamics, and generates novel physically-consistent trajectories.

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
* **Construct:** Embodied AI
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Decentralized Optimization with Topology-Independent Communication
**Date:** 2025-09-19 | **Arxiv:** [2509.14488](https://arxiv.org/abs/2509.14488)

#### Abstract
Distributed optimization requires nodes to coordinate, yet full synchronization scales poorly. When $n$ nodes collaborate through $m$ pairwise regularizers, standard methods demand $\mathcal{O}(m)$ communications per iteration. This paper proposes randomized local coordination: each node independently samples one regularizer uniformly and coordinates only with nodes sharing that term. This exploits partial separability, where each regularizer $G_j$ depends on a subset $S_j \subseteq \{1,\ldots,n\}$ of nodes. For graph-guided regularizers where $|S_j|=2$, expected communication drops to exactly 2 messages per iteration. This method achieves $\tilde{\mathcal{O}}(\varepsilon^{-2})$ iterations for convex objectives and under strong convexity, $\mathcal{O}(\varepsilon^{-1})$ to an $\varepsilon$-solution and $\mathcal{O}(\log(1/\varepsilon))$ to a neighborhood. Replacing the proximal map of the sum $\sum_j G_j$ with the proximal map of a single randomly selected regularizer $G_j$ preserves convergence while eliminating global coordination. Experiments validate both convergence rates and communication efficiency across synthetic and real-world datasets.

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


### Topology Structure Optimization of Reservoirs Using GLMY Homology
**Date:** 2025-09-16 | **Arxiv:** [2509.11612](https://arxiv.org/abs/2509.11612)

#### Abstract
Reservoir is an efficient network for time series processing. It is well known that network structure is one of the determinants of its performance. However, the topology structure of reservoirs, as well as their performance, is hard to analyzed, due to the lack of suitable mathematical tools. In this paper, we study the topology structure of reservoirs using persistent GLMY homology theory, and develop a method to improve its performance. Specifically, it is found that the reservoir performance is closely related to the one-dimensional GLMY homology groups. Then, we develop a reservoir structure optimization method by modifying the minimal representative cycles of one-dimensional GLMY homology groups. Finally, by experiments, it is validated that the performance of reservoirs is jointly influenced by the reservoir structure and the periodicity of the dataset.

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
* **Limits:** However, the topology structure of reservoirs, as well as their performance, is hard to analyzed, due to the lack of suitable mathematical tools.
* **Signal Tags:** #ai

---


### Feature Space Topology Control via Hopkins Loss
**Date:** 2025-09-16 | **Arxiv:** [2509.11154](https://arxiv.org/abs/2509.11154)

#### Abstract
Feature space topology refers to the organization of samples within the feature space. Modifying this topology can be beneficial in machine learning applications, including dimensionality reduction, generative modeling, transfer learning, and robustness to adversarial attacks. This paper introduces a novel loss function, Hopkins loss, which leverages the Hopkins statistic to enforce a desired feature space topology, which is in contrast to existing topology-related methods that aim to preserve input feature topology. We evaluate the effectiveness of Hopkins loss on speech, text, and image data in two scenarios: classification and dimensionality reduction using nonlinear bottleneck autoencoders. Our experiments show that integrating Hopkins loss into classification or dimensionality reduction has only a small impact on classification performance while providing the benefit of modifying feature topology.

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


### A Differential Manifold Perspective and Universality Analysis of Continuous Attractors in Artificial Neural Networks
**Date:** 2025-09-16 | **Arxiv:** [2509.10514](https://arxiv.org/abs/2509.10514)

#### Abstract
Continuous attractors are critical for information processing in both biological and artificial neural systems, with implications for spatial navigation, memory, and deep learning optimization. However, existing research lacks a unified framework to analyze their properties across diverse dynamical systems, limiting cross-architectural generalizability. This study establishes a novel framework from the perspective of differential manifolds to investigate continuous attractors in artificial neural networks. It verifies compatibility with prior conclusions, elucidates links between continuous attractor phenomena and eigenvalues of the local Jacobian matrix, and demonstrates the universality of singular value stratification in common classification models and datasets. These findings suggest continuous attractors may be ubiquitous in general neural networks, highlighting the need for a general theory, with the proposed framework offering a promising foundation given the close mathematical connection between eigenvalues and singular values.

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
* **Construct:** Embodied AI
* **Layer:** Infrastructure
* **Limits:** However, existing research lacks a unified framework to analyze their properties across diverse dynamical systems, limiting cross-architectural generalizability.
* **Signal Tags:** #ai

---


### Learning and Editing Universal Graph Prompt Tuning via Reinforcement Learning
**Date:** 2025-12-10 | **Arxiv:** [2512.08763](https://arxiv.org/abs/2512.08763)

#### Abstract
Early graph prompt tuning approaches relied on task-specific designs for Graph Neural Networks (GNNs), limiting their adaptability across diverse pre-training strategies. In contrast, another promising line of research has investigated universal graph prompt tuning, which operates directly in the input graph's feature space and builds a theoretical foundation that universal graph prompt tuning can theoretically achieve an equivalent effect of any prompting function, eliminating dependence on specific pre-training strategies. Recent works propose selective node-based graph prompt tuning to pursue more ideal prompts. However, we argue that selective node-based graph prompt tuning inevitably compromises the theoretical foundation of universal graph prompt tuning. In this paper, we strengthen the theoretical foundation of universal graph prompt tuning by introducing stricter constraints, demonstrating that adding prompts to all nodes is a necessary condition for achieving the universality of graph prompts. To this end, we propose a novel model and paradigm, Learning and Editing Universal GrAph Prompt Tuning (LEAP), which preserves the theoretical foundation of universal graph prompt tuning while pursuing more ideal prompts. Specifically, we first build the basic universal graph prompts to preserve the theoretical foundation and then employ actor-critic reinforcement learning to select nodes and edit prompts. Extensive experiments on graph- and node-level tasks across various pre-training strategies in both full-shot and few-shot scenarios show that LEAP consistently outperforms fine-tuning and other prompt-based approaches.

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
* **Limits:** However, we argue that selective node-based graph prompt tuning inevitably compromises the theoretical foundation of universal graph prompt tuning.
* **Signal Tags:** #ai

---


### PERM EQ x GRAPH EQ: Equivariant Neural Networks for Quantum Molecular Learning
**Date:** 2025-12-08 | **Arxiv:** [2512.05475](https://arxiv.org/abs/2512.05475)

#### Abstract
In hierarchal order of molecular geometry, we compare the performances of Geometric Quantum Machine Learning models. Two molecular datasets are considered: the simplistic linear shaped LiH-molecule and the trigonal pyramidal molecule NH3. Both accuracy and generalizability metrics are considered. A classical equivariant model is used as a baseline for the performance comparison. The comparative performance of Quantum Machine Learning models with no symmetry equivariance, rotational and permutational equivariance, and graph embedded permutational equivariance is investigated. The performance differentials and the molecular geometry in question reveals the criteria for choice of models for generalizability. Graph embedding of features is shown to be an effective pathway to greater trainability for geometric datasets. Permutational symmetric embedding is found to be the most generalizable quantum Machine Learning model for geometric learning.

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


### Statistical Inference for Manifold Similarity and Alignability across Noisy High-Dimensional Datasets
**Date:** 2025-11-27 | **Arxiv:** [2511.21074](https://arxiv.org/abs/2511.21074)

#### Abstract
The rapid growth of high-dimensional datasets across various scientific domains has created a pressing need for new statistical methods to compare distributions supported on their underlying structures. Assessing similarity between datasets whose samples lie on low-dimensional manifolds requires robust techniques capable of separating meaningful signal from noise. We propose a principled framework for statistical inference of similarity and alignment between distributions supported on manifolds underlying high-dimensional datasets in the presence of heterogeneous noise. The key idea is to link the low-rank structure of observed data matrices to their underlying manifold geometry. By analyzing the spectrum of the sample covariance under a manifold signal-plus-noise model, we develop a scale-invariant distance measure between datasets based on their principal variance structures. We further introduce a consistent estimator for this distance and a statistical test for manifold alignability, and establish their asymptotic properties using random matrix theory. The proposed framework accommodates heterogeneous noise across datasets and offers an efficient, theoretically grounded approach for comparing high-dimensional datasets with low-dimensional manifold structures. Through extensive simulations and analyses of multi-sample single-cell datasets, we demonstrate that our method achieves superior robustness and statistical power compared with existing approaches.

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


### Neural Tractability via Structure: Learning-Augmented Algorithms for Graph Combinatorial Optimization
**Date:** 2025-11-26 | **Arxiv:** [2511.19573](https://arxiv.org/abs/2511.19573)

#### Abstract
Neural models have shown promise in solving NP-hard graph combinatorial optimization (CO) problems. Once trained, they offer fast inference and reasonably high-quality solutions for in-distribution testing instances, but they generally fall short in terms of absolute solution quality compared to classical search-based algorithms that are admittedly slower but offer optimality guarantee once search finishes.   We propose a novel framework that combines the inference efficiency and exploratory power of neural models with the solution quality guarantee of search-based algorithms. In particular, we use parameterized algorithms (PAs) as the search component. PAs are dedicated to identifying easy instances of generally NP-hard problems, and allow for practically efficient search by exploiting structural simplicity (of the identified easy instances). Under our framework, we use parameterized analysis to identify the structurally hard parts of a CO instance. The neural model handles the hard parts by generating advisory signals based on its data-driven understanding. The PA-based search component then integrates the advisory signals to systematically and efficiently searches through the remaining structurally easy parts. Notably, our framework is agnostic to the choice of neural model and produces strictly better solutions than neural solvers alone.   We examine our framework on multiple CO tasks. Empirical results show that it achieves superior solution quality, competitive with that of commercial solvers. Furthermore, by using the neural model only for exploratory advisory signals, our framework exhibits improved out-of-distribution generalization, addressing a key limitation of existing neural CO solvers.

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


### Neurocircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification
**Date:** 2025-11-25 | **Arxiv:** [2511.17622](https://arxiv.org/abs/2511.17622)

#### Abstract
Major Depressive Disorder (MDD), affecting millions worldwide, exhibits complex pathophysiology manifested through disrupted brain network dynamics. Although graph neural networks that leverage neuroimaging data have shown promise in depression diagnosis, existing approaches are predominantly data-driven and operate largely as black-box models, lacking neurobiological interpretability. Here, we present NH-GCAT (Neurocircuitry-Inspired Hierarchical Graph Causal Attention Networks), a novel framework that bridges neuroscience domain knowledge with deep learning by explicitly and hierarchically modeling depression-specific mechanisms at different spatial scales. Our approach introduces three key technical contributions: (1) at the local brain regional level, we design a residual gated fusion module that integrates temporal blood oxygenation level dependent (BOLD) dynamics with functional connectivity patterns, specifically engineered to capture local depression-relevant low-frequency neural oscillations; (2) at the multi-regional circuit level, we propose a hierarchical circuit encoding scheme that aggregates regional node representations following established depression neurocircuitry organization, and (3) at the multi-circuit network level, we develop a variational latent causal attention mechanism that leverages a continuous probabilistic latent space to infer directed information flow among critical circuits, characterizing disease-altered whole-brain inter-circuit interactions. Rigorous leave-one-site-out cross-validation on the REST-meta-MDD dataset demonstrates NH-GCAT's state-of-the-art performance in depression classification, achieving a sample-size weighted-average accuracy of 73.3\% and an AUROC of 76.4\%, while simultaneously providing neurobiologically meaningful explanations.

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


### Graph Neural Networks vs Convolutional Neural Networks for Graph Domination Number Prediction
**Date:** 2025-11-25 | **Arxiv:** [2511.18150](https://arxiv.org/abs/2511.18150)

#### Abstract
We investigate machine learning approaches to approximating the \emph{domination number} of graphs, the minimum size of a dominating set. Exact computation of this parameter is NP-hard, restricting classical methods to small instances. We compare two neural paradigms: Convolutional Neural Networks (CNNs), which operate on adjacency matrix representations, and Graph Neural Networks (GNNs), which learn directly from graph structure through message passing. Across 2,000 random graphs with up to 64 vertices, GNNs achieve markedly higher accuracy ($R^2=0.987$, MAE $=0.372$) than CNNs ($R^2=0.955$, MAE $=0.500$). Both models offer substantial speedups over exact solvers, with GNNs delivering more than $200\times$ acceleration while retaining near-perfect fidelity. Our results position GNNs as a practical surrogate for combinatorial graph invariants, with implications for scalable graph optimization and mathematical discovery.

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


### LLM-Powered Text-Attributed Graph Anomaly Detection via Retrieval-Augmented Reasoning
**Date:** 2025-11-25 | **Arxiv:** [2511.17584](https://arxiv.org/abs/2511.17584)

#### Abstract
Anomaly detection on attributed graphs plays an essential role in applications such as fraud detection, intrusion monitoring, and misinformation analysis. However, text-attributed graphs (TAGs), in which node information is expressed in natural language, remain underexplored, largely due to the absence of standardized benchmark datasets. In this work, we introduce TAG-AD, a comprehensive benchmark for anomaly node detection on TAGs. TAG-AD leverages large language models (LLMs) to generate realistic anomalous node texts directly in the raw text space, producing anomalies that are semantically coherent yet contextually inconsistent and thus more reflective of real-world irregularities. In addition, TAG-AD incorporates multiple other anomaly types, enabling thorough and reproducible evaluation of graph anomaly detection (GAD) methods. With these datasets, we further benchmark existing unsupervised GNN-based GAD methods as well as zero-shot LLMs for GAD.   As part of our zero-shot detection setup, we propose a retrieval-augmented generation (RAG)-assisted, LLM-based zero-shot anomaly detection framework. The framework mitigates reliance on brittle, hand-crafted prompts by constructing a global anomaly knowledge base and distilling it into reusable analysis frameworks. Our experimental results reveal a clear division of strengths: LLMs are particularly effective at detecting contextual anomalies, whereas GNN-based methods remain superior for structural anomaly detection. Moreover, RAG-assisted prompting achieves performance comparable to human-designed prompts while eliminating manual prompt engineering, underscoring the practical value of our RAG-assisted zero-shot LLM anomaly detection framework.

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
* **Limits:** However, text-attributed graphs (TAGs), in which node information is expressed in natural language, remain underexplored, largely due to the absence of standardized benchmark datasets.
* **Signal Tags:** #ai

---


### ManifoldFormer: Geometric Deep Learning for Neural Dynamics on Riemannian Manifolds
**Date:** 2025-11-24 | **Arxiv:** [2511.16828](https://arxiv.org/abs/2511.16828)

#### Abstract
Existing EEG foundation models mainly treat neural signals as generic time series in Euclidean space, ignoring the intrinsic geometric structure of neural dynamics that constrains brain activity to low-dimensional manifolds. This fundamental mismatch between model assumptions and neural geometry limits representation quality and cross-subject generalization. ManifoldFormer addresses this limitation through a novel geometric deep learning framework that explicitly learns neural manifold representations. The architecture integrates three key innovations: a Riemannian VAE for manifold embedding that preserves geometric structure, a geometric Transformer with geodesic-aware attention mechanisms operating directly on neural manifolds, and a dynamics predictor leveraging neural ODEs for manifold-constrained temporal evolution. Extensive evaluation across four public datasets demonstrates substantial improvements over state-of-the-art methods, with 4.6-4.8% higher accuracy and 6.2-10.2% higher Cohen's Kappa, while maintaining robust cross-subject generalization. The geometric approach reveals meaningful neural patterns consistent with neurophysiological principles, establishing geometric constraints as essential for effective EEG foundation models.

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


### Reconstruction of Manifold Distances from Noisy Observations
**Date:** 2025-11-18 | **Arxiv:** [2511.13025](https://arxiv.org/abs/2511.13025)

#### Abstract
We consider the problem of reconstructing the intrinsic geometry of a manifold from noisy pairwise distance observations. Specifically, let $M$ denote a diameter 1 d-dimensional manifold and $μ$ a probability measure on $M$ that is mutually absolutely continuous with the volume measure. Suppose $X_1,\dots,X_N$ are i.i.d. samples of $μ$ and we observe noisy-distance random variables $d'(X_j, X_k)$ that are related to the true geodesic distances $d(X_j,X_k)$. With mild assumptions on the distributions and independence of the noisy distances, we develop a new framework for recovering all distances between points in a sufficiently dense subsample of $M$. Our framework improves on previous work which assumed i.i.d. additive noise with known moments. Our method is based on a new way to estimate $L_2$-norms of certain expectation-functions $f_x(y)=\mathbb{E}d'(x,y)$ and use them to build robust clusters centered at points of our sample. Using a new geometric argument, we establish that, under mild geometric assumptions--bounded curvature and positive injectivity radius--these clusters allow one to recover the true distances between points in the sample up to an additive error of $O(\varepsilon \log \varepsilon^{-1})$. We develop two distinct algorithms for producing these clusters. The first achieves a sample complexity $N \asymp \varepsilon^{-2d-2}\log(1/\varepsilon)$ and runtime $o(N^3)$. The second introduces novel geometric ideas that warrant further investigation. In the presence of missing observations, we show that a quantitative lower bound on sampling probabilities suffices to modify the cluster construction in the first algorithm and extend all recovery guarantees. Our main technical result also elucidates which properties of a manifold are necessary for the distance recovery, which suggests further extension of our techniques to a broader class of metric probability spaces.

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


### Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs
**Date:** 2025-11-11 | **Arxiv:** [2511.07419](https://arxiv.org/abs/2511.07419)

#### Abstract
Sparse Mixture-of-Experts (MoE) have been widely adopted in recent large language models since it can efficiently scale up the model capability without increasing the inference cost. However, evaluations on broad downstream tasks reveal a consistent suboptimality of the routers in existing MoE LLMs, which results in a severe performance gap (e.g., 10-20% in accuracy) to the optimal routing. In this paper, we show that aligning the manifold of routing weights with that of task embedding can effectively reduce the gap and improve MoE LLMs' generalization performance. Our method, "Routing Manifold Alignment (RoMA)", introduces an additional manifold regularization term in the post-training objective and only requires lightweight finetuning of routers (with other parameters frozen). Specifically, the regularization encourages the routing weights of each sample to be close to those of its successful neighbors (whose routing weights lead to correct answers) in a task embedding space. Consequently, samples targeting similar tasks will share similar expert choices across layers. Building such bindings between tasks and experts over different samples is essential to achieve better generalization. Moreover, RoMA demonstrates the advantage of unifying the task understanding (by embedding models) with solution generation (by MoE LLMs). In experiments, we finetune routers in OLMoE, DeepSeekMoE, and Qwen3-MoE using RoMA. Evaluations on diverse benchmarks and extensive comparisons with baselines show the substantial improvement brought by RoMA.

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
* **Limits:** However, evaluations on broad downstream tasks reveal a consistent suboptimality of the routers in existing MoE LLMs, which results in a severe performance gap (e.
* **Signal Tags:** #ai

---


### Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning
**Date:** 2025-11-11 | **Arxiv:** [2511.06946](https://arxiv.org/abs/2511.06946)

#### Abstract
Transformers have shown strong ability to model long-term dependencies and are increasingly adopted as world models in model-based reinforcement learning (RL) under partial observability. However, unlike natural language corpora, RL trajectories are sparse and reward-driven, making standard self-attention inefficient because it distributes weight uniformly across all past tokens rather than emphasizing the few transitions critical for control. To address this, we introduce structured inductive priors into the self-attention mechanism of the dynamics head: (i) per-head memory-length priors that constrain attention to task-specific windows, and (ii) distributional priors that learn smooth Gaussian weightings over past state-action pairs. We integrate these mechanisms into UniZero, a model-based RL agent with a Transformer-based world model that supports planning under partial observability. Experiments on the Atari 100k benchmark show that most efficiency gains arise from the Gaussian prior, which smoothly allocates attention to informative transitions, while memory-length priors often truncate useful signals with overly restrictive cut-offs. In particular, Gaussian Attention achieves a 77% relative improvement in mean human-normalized scores over UniZero. These findings suggest that in partially observable RL domains with non-stationary temporal dependencies, discrete memory windows are difficult to learn reliably, whereas smooth distributional priors flexibly adapt across horizons and yield more robust data efficiency. Overall, our results demonstrate that encoding structured temporal priors directly into self-attention improves the prioritization of informative histories for dynamics modeling under partial observability.

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
* **Limits:** However, unlike natural language corpora, RL trajectories are sparse and reward-driven, making standard self-attention inefficient because it distributes weight uniformly across all past tokens rather than emphasizing the few transitions critical for control.
* **Signal Tags:** #ai

---


### Optimization without Retraction on the Random Generalized Stiefel Manifold
**Date:** 2025-11-11 | **Arxiv:** [2405.01702](https://arxiv.org/abs/2405.01702)

#### Abstract
Optimization over the set of matrices $X$ that satisfy $X^\top B X = I_p$, referred to as the generalized Stiefel manifold, appears in many applications involving sampled covariance matrices such as the canonical correlation analysis (CCA), independent component analysis (ICA), and the generalized eigenvalue problem (GEVP). Solving these problems is typically done by iterative methods that require a fully formed $B$. We propose a cheap stochastic iterative method that solves the optimization problem while having access only to random estimates of $B$. Our method does not enforce the constraint in every iteration; instead, it produces iterations that converge to critical points on the generalized Stiefel manifold defined in expectation. The method has lower per-iteration cost, requires only matrix multiplications, and has the same convergence rates as its Riemannian optimization counterparts that require the full matrix $B$. Experiments demonstrate its effectiveness in various machine learning applications involving generalized orthogonality constraints, including CCA, ICA, and the GEVP.

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


### Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks
**Date:** 2025-11-07 | **Arxiv:** [2511.03824](https://arxiv.org/abs/2511.03824)

#### Abstract
Graph Neural Networks learn on graph-structured data by iteratively aggregating local neighborhood information. While this local message passing paradigm imparts a powerful inductive bias and exploits graph sparsity, it also yields three key challenges: (i) oversquashing of long-range information, (ii) oversmoothing of node representations, and (iii) limited expressive power. In this work we inject randomized global embeddings of node features, which we term \textit{Sketched Random Features}, into standard GNNs, enabling them to efficiently capture long-range dependencies. The embeddings are unique, distance-sensitive, and topology-agnostic -- properties which we analytically and empirically show alleviate the aforementioned limitations when injected into GNNs. Experimental results on real-world graph learning tasks confirm that this strategy consistently improves performance over baseline GNNs, offering both a standalone solution and a complementary enhancement to existing techniques such as graph positional encodings. Our source code is available at \href{https://github.com/ryienh/sketched-random-features}{https://github.com/ryienh/sketched-random-features}.

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


### Dynamical loss functions shape landscape topography and improve learning in artificial neural networks
**Date:** 2025-11-06 | **Arxiv:** [2410.10690](https://arxiv.org/abs/2410.10690)

#### Abstract
Dynamical loss functions are derived from standard loss functions used in supervised classification tasks, but are modified so that the contribution from each class periodically increases and decreases. These oscillations globally alter the loss landscape without affecting the global minima. In this paper, we demonstrate how to transform cross-entropy and mean squared error into dynamical loss functions. We begin by discussing the impact of increasing the size of the neural network or the learning rate on the depth and sharpness of the minima that the system explores. Building on this intuition, we propose several versions of dynamical loss functions and use a simple classification problem where we can show how they significantly improve validation accuracy for networks of varying sizes. Finally, we explore how the landscape of these dynamical loss functions evolves during training, highlighting the emergence of instabilities that may be linked to edge-of-instability minimization.

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


### Theoretical Guarantees for Causal Discovery on Large Random Graphs
**Date:** 2025-11-05 | **Arxiv:** [2511.02536](https://arxiv.org/abs/2511.02536)

#### Abstract
We investigate theoretical guarantees for the false-negative rate (FNR) -- the fraction of true causal edges whose orientation is not recovered, under single-variable random interventions and an $ε$-interventional faithfulness assumption that accommodates latent confounding. For sparse Erdős--Rényi directed acyclic graphs, where the edge probability scales as $p_e = Θ(1/d)$, we show that the FNR concentrates around its mean at rate $O(\frac{\log d}{\sqrt d})$, implying that large deviations above the expected error become exponentially unlikely as dimensionality increases. This concentration ensures that derived upper bounds hold with high probability in large-scale settings. Extending the analysis to generalized Barabási--Albert graphs reveals an even stronger phenomenon: when the degree exponent satisfies $γ> 3$, the deviation width scales as $O(d^{β- \frac{1}{2}})$ with $β= 1/(γ- 1) < \frac{1}{2}$, and hence vanishes in the limit. This demonstrates that realistic scale-free topologies intrinsically regularize causal discovery, reducing variability in orientation error. These finite-dimension results provide the first dimension-adaptive, faithfulness-robust guarantees for causal structure recovery, and challenge the intuition that high dimensionality and network heterogeneity necessarily hinder accurate discovery. Our simulation results corroborate these theoretical predictions, showing that the FNR indeed concentrates and often vanishes in practice as dimensionality grows.

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


### Disentangling Causal Substructures for Interpretable and Generalizable Drug Synergy Prediction
**Date:** 2025-11-05 | **Arxiv:** [2511.02146](https://arxiv.org/abs/2511.02146)

#### Abstract
Drug synergy prediction is a critical task in the development of effective combination therapies for complex diseases, including cancer. Although existing methods have shown promising results, they often operate as black-box predictors that rely predominantly on statistical correlations between drug characteristics and results. To address this limitation, we propose CausalDDS, a novel framework that disentangles drug molecules into causal and spurious substructures, utilizing the causal substructure representations for predicting drug synergy. By focusing on causal sub-structures, CausalDDS effectively mitigates the impact of redundant features introduced by spurious substructures, enhancing the accuracy and interpretability of the model. In addition, CausalDDS employs a conditional intervention mechanism, where interventions are conditioned on paired molecular structures, and introduces a novel optimization objective guided by the principles of sufficiency and independence. Extensive experiments demonstrate that our method outperforms baseline models, particularly in cold start and out-of-distribution settings. Besides, CausalDDS effectively identifies key substructures underlying drug synergy, providing clear insights into how drug combinations work at the molecular level. These results underscore the potential of CausalDDS as a practical tool for predicting drug synergy and facilitating drug discovery.

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


### Assessing LLM Reasoning Steps via Principal Knowledge Grounding
**Date:** 2025-11-04 | **Arxiv:** [2511.00879](https://arxiv.org/abs/2511.00879)

#### Abstract
Step-by-step reasoning has become a standard approach for large language models (LLMs) to tackle complex tasks. While this paradigm has proven effective, it raises a fundamental question: How can we verify that an LLM's reasoning is accurately grounded in knowledge? To address this question, we introduce a novel evaluation suite that systematically assesses the knowledge grounding of intermediate reasoning. Our framework comprises three key components. (1) Principal Knowledge Collection, a large-scale repository of atomic knowledge essential for reasoning. Based on the collection, we propose (2) knowledge-grounded evaluation metrics designed to measure how well models recall and apply prerequisite knowledge in reasoning. These metrics are computed by our (3) evaluator LLM, a lightweight model optimized for cost-effective and reliable metric computation. Our evaluation suite demonstrates remarkable effectiveness in identifying missing or misapplied knowledge elements, providing crucial insights for uncovering fundamental reasoning deficiencies in LLMs. Beyond evaluation, we demonstrate how these metrics can be integrated into preference optimization, showcasing further applications of knowledge-grounded evaluation.

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
