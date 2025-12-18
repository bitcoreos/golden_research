# Vol 21 Efficiency   Quantization
*Enriched by BITCOREOS | Phase 4 Batch 5*

---

### Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning
**Date:** 2025-10-17 | **Arxiv:** [2510.13832](https://hub.bitwiki.org/t/entropy-meets-importance-a-unified-head-importance-entropy-score-for-stable-and-efficient-transformer-pruning/17529)

#### Abstract
Transformer-based models have achieved remarkable performance in NLP tasks. However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment. To address these challenges, various pruning methods have recently been proposed. Notably, gradient-based methods using Head Importance Scores (HIS) have gained traction for interpretability, efficiency, and ability to identify redundant heads. However, HIS alone has limitations as it captures only the gradient-driven contribution, overlooking the diversity of attention patterns. To overcome these limitations, we introduce a novel pruning criterion, HIES (Head Importance-Entropy Score), which integrates head importance scores with attention entropy, providing complementary evidence on per-head contribution. Empirically, HIES-based pruning yields up to 15.2% improvement in model quality and 2.04x improvement in stability over HIS-only methods, enabling substantial model compression without sacrificing either accuracy or stability. Code will be released upon publication.

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
* **Limits:** However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment.
* **Signal Tags:** #ai

---


### The Impact of Quantization on Large Reasoning Model Reinforcement Learning
**Date:** 2025-11-20 | **Arxiv:** [2511.15694](https://hub.bitwiki.org/t/the-impact-of-quantization-on-large-reasoning-model-reinforcement-learning/24808)

#### Abstract
Strong reasoning capabilities can now be achieved by large-scale reinforcement learning (RL) without any supervised fine-tuning. Although post-training quantization (PTQ) and quantization-aware training (QAT) are well studied in the context of fine-tuning, how quantization impacts RL in large reasoning models (LRMs) remains an open question. To answer this question, we conducted systematic experiments and discovered a significant gap in reasoning performance on mathematical benchmarks between post-RL quantized models and their quantization-aware RL optimized counterparts. Our findings suggest that quantization-aware RL training negatively impacted the learning process, whereas PTQ and QLoRA led to greater performance.

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


### Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning
**Date:** 2025-10-23 | **Arxiv:** [2510.19338](https://hub.bitwiki.org/t/every-attention-matters-an-efficient-hybrid-architecture-for-long-context-reasoning/18876)

#### Abstract
In this technical report, we present the Ring-linear model series, specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0. Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both models adopt a hybrid architecture that effectively integrates linear attention and softmax attention, significantly reducing I/O and computational overhead in long-context inference scenarios. Compared to a 32 billion parameter dense model, this series reduces inference cost to 1/10, and compared to the original Ring series, the cost is also reduced by over 50%. Furthermore, through systematic exploration of the ratio between different attention mechanisms in the hybrid architecture, we have identified the currently optimal model structure. Additionally, by leveraging our self-developed high-performance FP8 operator library-linghe, overall training efficiency has been improved by 50%. Benefiting from the high alignment between the training and inference engine operators, the models can undergo long-term, stable, and highly efficient optimization during the reinforcement learning phase, consistently maintaining SOTA performance across multiple challenging complex reasoning benchmarks.

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


### Efficient Few-Shot Learning in Remote Sensing: Fusing Vision and Vision-Language Models
**Date:** 2025-10-17 | **Arxiv:** [2510.13993](https://hub.bitwiki.org/t/efficient-few-shot-learning-in-remote-sensing-fusing-vision-and-vision-language-models/17545)

#### Abstract
Remote sensing has become a vital tool across sectors such as urban planning, environmental monitoring, and disaster response. While the volume of data generated has increased significantly, traditional vision models are often constrained by the requirement for extensive domain-specific labelled data and their limited ability to understand the context within complex environments. Vision Language Models offer a complementary approach by integrating visual and textual data; however, their application to remote sensing remains underexplored, particularly given their generalist nature. This work investigates the combination of vision models and VLMs to enhance image analysis in remote sensing, with a focus on aircraft detection and scene understanding. The integration of YOLO with VLMs such as LLaVA, ChatGPT, and Gemini aims to achieve more accurate and contextually aware image interpretation. Performance is evaluated on both labelled and unlabelled remote sensing data, as well as degraded image scenarios which are crucial for remote sensing. The findings show an average MAE improvement of 48.46% across models in the accuracy of aircraft detection and counting, especially in challenging conditions, in both raw and degraded scenarios. A 6.17% improvement in CLIPScore for comprehensive understanding of remote sensing images is obtained. The proposed approach combining traditional vision models and VLMs paves the way for more advanced and efficient remote sensing image analysis, especially in few-shot learning scenarios.

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
* **Limits:** however, their application to remote sensing remains underexplored, particularly given their generalist nature.
* **Signal Tags:** #ai

---


### SPARTAN: A Sparse Transformer World Model Attending to What Matters
**Date:** 2025-12-08 | **Arxiv:** [2411.06890](https://hub.bitwiki.org/t/spartan-a-sparse-transformer-world-model-attending-to-what-matters/27867)

#### Abstract
Capturing the interactions between entities in a structured way plays a central role in world models that flexibly adapt to changes in the environment. Recent works motivate the benefits of models that explicitly represent the structure of interactions and formulate the problem as discovering local causal structures. In this work, we demonstrate that reliably capturing these relationships in complex settings remains challenging. To remedy this shortcoming, we postulate that sparsity is a critical ingredient for the discovery of such local structures. To this end, we present the SPARse TrANsformer World model (SPARTAN), a Transformer-based world model that learns context-dependent interaction structures between entities in a scene. By applying sparsity regularisation on the attention patterns between object-factored tokens, SPARTAN learns sparse, context-dependent interaction graphs that accurately predict future object states. We further extend our model to adapt to sparse interventions with unknown targets in the dynamics of the environment. This results in a highly interpretable world model that can efficiently adapt to changes. Empirically, we evaluate SPARTAN against the current state-of-the-art in object-centric world models in observation-based environments and demonstrate that our model can learn local causal graphs that accurately reflect the underlying interactions between objects, achieving significantly improved few-shot adaptation to dynamics changes, as well as robustness against distractors.

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


### A note on the impossibility of conditional PAC-efficient reasoning in large language models
**Date:** 2025-12-04 | **Arxiv:** [2512.03057](https://hub.bitwiki.org/t/a-note-on-the-impossibility-of-conditional-pac-efficient-reasoning-in-large-language-models/27473)

#### Abstract
We prove an impossibility result for conditional Probably Approximately Correct (PAC)-efficient reasoning in large language models. While recent work has established marginal PAC efficiency guarantees for composite models that switch between expensive expert models and cheaper fast models, we show that conditional (pointwise) guarantees are impossible in the distribution-free setting. Specifically, for non-atomic input spaces, any algorithm achieving conditional PAC efficiency must be trivial in the sense that it defers to the expert model with probability at least $1-α$ for almost every input.

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


### Model-to-Model Knowledge Transmission (M2KT): A Data-Free Framework for Cross-Model Understanding Transfer
**Date:** 2025-11-25 | **Arxiv:** [2511.17638](https://hub.bitwiki.org/t/model-to-model-knowledge-transmission-m2kt-a-data-free-framework-for-cross-model-understanding-transfer/25369)

#### Abstract
Modern artificial intelligence systems depend heavily on large datasets for both training and transferring knowledge between models. Knowledge distillation, transfer learning, and dataset distillation have made such transfers more efficient, yet they remain fundamentally data-driven: a teacher must produce examples, logits, or gradients for a student to learn. In this work, we introduce Model-to-Model Knowledge Transmission (M2KT), a novel paradigm for data-free conceptual transfer between neural networks. M2KT enables models to exchange knowledge packets that encapsulate structured concept embeddings, abstraction graphs, reasoning traces, and provenance metadata. Unlike classical distillation, M2KT operates primarily in concept space rather than example space, and it does not require labeled datasets or teacher-generated outputs during transfer. We formalize the notion of concept manifolds, introduce an inter-model alignment mapping between teacher and student latent spaces, and derive a composite loss that enforces geometric, structural, and reasoning consistency together with explicit safety constraints. We further present algorithmic procedures for teacher-side packet generation and student-side ingestion and verification. Experiments on symbolic reasoning with large language models show that M2KT can achieve approximately 85 to 90 percent of teacher performance while reducing data usage by over 98 percent compared to standard knowledge distillation. This work establishes a theoretical and practical foundation for data-free AI-to-AI knowledge transfer and self-improving model ecosystems.

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


### Optimal Self-Consistency for Efficient Reasoning with Large Language Models
**Date:** 2025-11-18 | **Arxiv:** [2511.12309](https://hub.bitwiki.org/t/optimal-self-consistency-for-efficient-reasoning-with-large-language-models/23953)

#### Abstract
Self-consistency (SC) is a widely used test-time inference technique for improving performance in chain-of-thought reasoning. It involves generating multiple responses, or samples from a large language model (LLM) and selecting the most frequent answer. This procedure can naturally be viewed as a majority vote or empirical mode estimation. Despite its effectiveness, SC is prohibitively expensive at scale when naively applied to datasets, and it lacks a unified theoretical treatment of sample efficiency and scaling behavior. In this paper, we provide the first comprehensive analysis of SC's scaling behavior and its variants, drawing on mode estimation and voting theory. We derive and empirically validate power law scaling for self-consistency across datasets, and analyze the sample efficiency for fixed-allocation and dynamic-allocation sampling schemes. From these insights, we introduce Blend-ASC, a novel variant of self-consistency that dynamically allocates samples to questions during inference, achieving state-of-the-art sample efficiency. Our approach uses 6.8x fewer samples than vanilla SC on average, outperforming both fixed- and dynamic-allocation SC baselines, thereby demonstrating the superiority of our approach in terms of efficiency. In contrast to existing variants, Blend-ASC is hyperparameter-free and can fit an arbitrary sample budget, ensuring it can be easily applied to any self-consistency application.

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


### Know Your Limits: Entropy Estimation Modeling for Compression and Generalization
**Date:** 2025-11-14 | **Arxiv:** [2511.10618](https://hub.bitwiki.org/t/know-your-limits-entropy-estimation-modeling-for-compression-and-generalization/23586)

#### Abstract
Language prediction is constrained by informational entropy intrinsic to language, such that there exists a limit to how accurate any language model can become and equivalently a lower bound to language compression. The most efficient language compression algorithms today are causal (next token prediction) large language models, but the use of these models to form accurate estimates of language entropy is currently computationally infeasible. We introduce encoder-augmented causal decoder model architectures that exhibit superior training efficiency characteristics and achieve higher compression than causal transformers even when trained on modest hardware. We demonstrate how entropy estimates can be obtained on a per-token basis, and show that the generalization of models trained to approach the entropy of their training data necessarily exceeds the generalization of models trained to minimize loss beyond this value. We show empirically that causal models trained to approach but not exceed estimated per-token entropies exhibit greater generalization than models trained without taking entropy into account.

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


### Federated Learning for Video Violence Detection: Complementary Roles of Lightweight CNNs and Vision-Language Models for Energy-Efficient Use
**Date:** 2025-11-11 | **Arxiv:** [2511.07171](https://hub.bitwiki.org/t/federated-learning-for-video-violence-detection-complementary-roles-of-lightweight-cnns-and-vision-language-models-for-energy-efficient-use/22796)

#### Abstract
Deep learning-based video surveillance increasingly demands privacy-preserving architectures with low computational and environmental overhead. Federated learning preserves privacy but deploying large vision-language models (VLMs) introduces major energy and sustainability challenges. We compare three strategies for federated violence detection under realistic non-IID splits on the RWF-2000 and RLVS datasets: zero-shot inference with pretrained VLMs, LoRA-based fine-tuning of LLaVA-NeXT-Video-7B, and personalized federated learning of a 65.8M-parameter 3D CNN. All methods exceed 90% accuracy in binary violence detection. The 3D CNN achieves superior calibration (ROC AUC 92.59%) at roughly half the energy cost (240 Wh vs. 570 Wh) of federated LoRA, while VLMs provide richer multimodal reasoning. Hierarchical category grouping (based on semantic similarity and class exclusion) boosts VLM multiclass accuracy from 65.31% to 81% on the UCF-Crime dataset. To our knowledge, this is the first comparative simulation study of LoRA-tuned VLMs and personalized CNNs for federated violence detection, with explicit energy and CO2e quantification. Our results inform hybrid deployment strategies that default to efficient CNNs for routine inference and selectively engage VLMs for complex contextual reasoning.

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


### ProofSketch: Efficient Verified Reasoning for Large Language Models
**Date:** 2025-10-30 | **Arxiv:** [2510.24811](https://hub.bitwiki.org/t/proofsketch-efficient-verified-reasoning-for-large-language-models/20362)

#### Abstract
Reasoning methods such as chain-of-thought prompting and self-consistency have shown immense potential to improve the accuracy of large language models across various reasoning tasks. However such methods involve generation of lengthy reasoning chains, which substantially increases token consumption, computational cost, and latency. To address this inefficiency, we propose ProofSketch, a verification-guided reasoning framework that integrates symbolic closure computation, lexicographic verification and adaptive sketch generation. Our experiments show that ProofSketch consistently reduces token usage while improving accuracy, demonstrating that this approach offers a promising path for efficient and trustworthy reasoning.

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


### Low Power Vision Transformer Accelerator with Hardware-Aware Pruning and Optimized Dataflow
**Date:** 2025-10-17 | **Arxiv:** [2510.14393](https://hub.bitwiki.org/t/low-power-vision-transformer-accelerator-with-hardware-aware-pruning-and-optimized-dataflow/17566)

#### Abstract
Current transformer accelerators primarily focus on optimizing self-attention due to its quadratic complexity. However, this focus is less relevant for vision transformers with short token lengths, where the Feed-Forward Network (FFN) tends to be the dominant computational bottleneck. This paper presents a low power Vision Transformer accelerator, optimized through algorithm-hardware co-design. The model complexity is reduced using hardware-friendly dynamic token pruning without introducing complex mechanisms. Sparsity is further improved by replacing GELU with ReLU activations and employing dynamic FFN2 pruning, achieving a 61.5\% reduction in operations and a 59.3\% reduction in FFN2 weights, with an accuracy loss of less than 2\%. The hardware adopts a row-wise dataflow with output-oriented data access to eliminate data transposition, and supports dynamic operations with minimal area overhead. Implemented in TSMC's 28nm CMOS technology, our design occupies 496.4K gates and includes a 232KB SRAM buffer, achieving a peak throughput of 1024 GOPS at 1GHz, with an energy efficiency of 2.31 TOPS/W and an area efficiency of 858.61 GOPS/mm2.

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
* **Layer:** Hardware
* **Limits:** However, this focus is less relevant for vision transformers with short token lengths, where the Feed-Forward Network (FFN) tends to be the dominant computational bottleneck.
* **Signal Tags:** #ai

---


### The Inhibitor: ReLU and Addition-Based Attention for Efficient Transformers under Fully Homomorphic Encryption on the Torus
**Date:** 2025-10-02 | **Arxiv:** [2310.02041](https://hub.bitwiki.org/t/the-inhibitor-relu-and-addition-based-attention-for-efficient-transformers-under-fully-homomorphic-encryption-on-the-torus/13928)

#### Abstract
To enhance the computational efficiency of quantized Transformers, we replace the dot-product and Softmax-based attention with an alternative mechanism involving addition and ReLU activation only. This side-steps the expansion to double precision often required by matrix multiplication and avoids costly Softmax evaluations but maintains much of the core functionality of conventional dot-product attention. It can enable more efficient execution and support larger quantized Transformer models on resource-constrained hardware or alternative arithmetic systems like homomorphic encryption. Training experiments on four common benchmark tasks show test set prediction scores comparable to those of conventional Transformers with dot-product attention. Our scaling experiments also suggest significant computational savings, both in plaintext and under encryption. In particular, we believe that the ReLU and addition-based attention mechanism examined in this paper may enable privacy-preserving AI applications operating under homomorphic encryption by avoiding the costly multiplication of encrypted variables.

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


### PCoreSet: Effective Active Learning through Knowledge Distillation from Vision-Language Models
**Date:** 2025-10-02 | **Arxiv:** [2506.00910](https://hub.bitwiki.org/t/pcoreset-effective-active-learning-through-knowledge-distillation-from-vision-language-models/13948)

#### Abstract
Knowledge distillation (KD) is a widely used framework for training compact, task-specific models by transferring the knowledge from teacher models. However, its application to active learning (AL), which aims to minimize annotation costs through iterative sample selection, remains underexplored. This gap stems from the fact that KD typically assumes access to sufficient labeled data, whereas AL operates in data-scarce scenarios where task-specific teacher models are often unavailable. In this paper, we first introduce ActiveKD, a framework that integrates AL with KD by leveraging the zero- and few-shot capabilities of large vision-language models (VLMs). A key aspect of ActiveKD is the structured prediction bias of VLMs-i.e., their predictions form clusters in the probability space. We regard this structure as an inductive bias of the teacher model, capturing generalizable output patterns beneficial to student learning. To exploit this bias, we propose Probabilistic CoreSet (PCoreSet), a selection strategy that maximizes coverage in the probability space rather than the feature space. PCoreSet strategically selects probabilistically diverse unlabeled samples, facilitating more efficient transfer of teacher knowledge under limited annotation budgets. Extensive evaluations on 11 datasets show that ActiveKD consistently improves performance across selection methods (e.g., +29.07% on ImageNet, averaged over methods). Under ActiveKD, PCoreSet ranks first in 64/73 settings (approximately 87.7%) across 5 student and 3 teacher networks, always achieving the best performance except for first 2 AL rounds. Our code is available at https://github.com/erjui/PCoreSet.

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
* **Layer:** Application
* **Limits:** However, its application to active learning (AL), which aims to minimize annotation costs through iterative sample selection, remains underexplored.
* **Signal Tags:** #ai

---


### DaMoC: Efficiently Selecting the Optimal Large Language Model for Fine-tuning Domain Tasks Based on Data and Model Compression
**Date:** 2025-09-03 | **Arxiv:** [2509.01221](https://hub.bitwiki.org/t/damoc-efficiently-selecting-the-optimal-large-language-model-for-fine-tuning-domain-tasks-based-on-data-and-model-compression/7339)

#### Abstract
Large language models (LLMs) excel in general tasks but struggle with domain-specific ones, requiring fine-tuning with specific data. With many open-source LLMs available, selecting the best model for fine-tuning downstream tasks is challenging, primarily focusing on how to quickly identify the optimal LLM. We introduce a Data and Model Compression Framework (DaMoC) that addresses this challenge by: 1) Data Level: A systematic categorization of data filtering methodologies for LLMs is first established, classifying them into three distinct paradigms: (1) distribution-aware methods, (2) quality-aware methods, and (3) hybrid approaches considering both dimensions. Further, we enhance the density of key tokens in the text achieving token compression. Subsequently, we use an LLM to iterative rewrite the text to optimize its expression. 2) Model Level: We use layer similarity scores to assess each layer's importance and remove those with lower importance. Then, we introduce a sparse merging paradigm to preserve as much of the original model's capability as possible. Extensive experiments on four datasets, medical Q&A, financial Q&A, general Q&A, and reading comprehension, show that we can select the optimal LLM while saving approximately 20-fold in training time.

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


### NEAT: Neighborhood-Guided, Efficient, Autoregressive Set Transformer for 3D Molecular Generation
**Date:** 2025-12-08 | **Arxiv:** [2512.05844](https://hub.bitwiki.org/t/neat-neighborhood-guided-efficient-autoregressive-set-transformer-for-3d-molecular-generation/27893)

#### Abstract
Autoregressive models are a promising alternative to diffusion-based models for 3D molecular structure generation. However, a key limitation is the assumption of a token order: while text has a natural sequential order, the next token prediction given a molecular graph prefix should be invariant to atom permutations. Previous works sidestepped this mismatch by using canonical orders or focus atoms. We argue that this is unnecessary. We introduce NEAT, a Neighborhood-guided, Efficient, Autoregressive, Set Transformer that treats molecular graphs as sets of atoms and learns the order-agnostic distribution over admissible tokens at the graph boundary with an autoregressive flow model. NEAT approaches state-of-the-art performance in 3D molecular generation with high computational efficiency and atom-level permutation invariance, establishing a practical foundation for scalable molecular design.

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
* **Limits:** However, a key limitation is the assumption of a token order: while text has a natural sequential order, the next token prediction given a molecular graph prefix should be invariant to atom permutations.
* **Signal Tags:** #ai

---


### Efficient Generative Transformer Operators For Million-Point PDEs
**Date:** 2025-12-05 | **Arxiv:** [2512.04974](https://hub.bitwiki.org/t/efficient-generative-transformer-operators-for-million-point-pdes/27668)

#### Abstract
We introduce ECHO, a transformer-operator framework for generating million-point PDE trajectories. While existing neural operators (NOs) have shown promise for solving partial differential equations, they remain limited in practice due to poor scalability on dense grids, error accumulation during dynamic unrolling, and task-specific design. ECHO addresses these challenges through three key innovations. (i) It employs a hierarchical convolutional encode-decode architecture that achieves a 100 $\times$ spatio-temporal compression while preserving fidelity on mesh points. (ii) It incorporates a training and adaptation strategy that enables high-resolution PDE solution generation from sparse input grids. (iii) It adopts a generative modeling paradigm that learns complete trajectory segments, mitigating long-horizon error drift. The training strategy decouples representation learning from downstream task supervision, allowing the model to tackle multiple tasks such as trajectory generation, forward and inverse problems, and interpolation. The generative model further supports both conditional and unconditional generation. We demonstrate state-of-the-art performance on million-point simulations across diverse PDE systems featuring complex geometries, high-frequency dynamics, and long-term horizons.

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


### Efficient Mathematical Reasoning Models via Dynamic Pruning and Knowledge Distillation
**Date:** 2025-11-25 | **Arxiv:** [2511.17577](https://hub.bitwiki.org/t/efficient-mathematical-reasoning-models-via-dynamic-pruning-and-knowledge-distillation/25306)

#### Abstract
With the rapid development of deep learning, large language models have shown strong capabilities in complex reasoning tasks such as mathematical equation solving. However, their substantial computational and storage costs hinder practical deployment. This paper proposes a lightweight optimization method that integrates dynamic attention head pruning with knowledge distillation. The approach dynamically evaluates the importance of each attention head in the multi-head attention mechanism using a combination of weight norms and entropy, and prunes redundant heads in real time to reduce computational overhead. To mitigate performance degradation, knowledge distillation transfers information from the original model to the pruned student, enabling the smaller model to preserve reasoning ability. Experiments conducted on both Math23k and ASDiv-A verify the effectiveness of the proposed method. For example, on Math23k with a 30% pruning ratio, parameters are reduced by 18.7%, inference speed is improved by 27.5%, FLOPs are reduced by 19.3%, and accuracy drops only 0.7% (from 84.4% to 83.7%). These results demonstrate that the method achieves substantial efficiency gains while maintaining strong reasoning performance, providing a practical solution for efficient deployment of large language models in mathematical reasoning tasks.

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
* **Limits:** However, their substantial computational and storage costs hinder practical deployment.
* **Signal Tags:** #ai

---


### Sometimes Painful but Certainly Promising: Feasibility and Trade-offs of Language Model Inference at the Edge
**Date:** 2025-11-24 | **Arxiv:** [2503.09114](https://hub.bitwiki.org/t/sometimes-painful-but-certainly-promising-feasibility-and-trade-offs-of-language-model-inference-at-the-edge/25259)

#### Abstract
The rapid rise of Language Models (LMs) has expanded the capabilities of natural language processing, powering applications from text generation to complex decision-making. While state-of-the-art LMs often boast hundreds of billions of parameters and are primarily deployed in data centers, recent trends show a growing focus on compact models-typically under 10 billion parameters-enabled by techniques such as quantization and other model compression techniques. This shift paves the way for LMs on edge devices, offering potential benefits such as enhanced privacy, reduced latency, and improved data sovereignty. However, the inherent complexity of even these smaller models, combined with the limited computing resources of edge hardware, raises critical questions about the practical trade-offs in executing LM inference outside the cloud. To address these challenges, we present a comprehensive evaluation of generative LM inference on representative CPU-based and GPU-accelerated edge devices. Our study measures key performance indicators-including memory usage, inference speed, and energy consumption-across various device configurations. Additionally, we examine throughput-energy trade-offs, cost considerations, and usability, alongside an assessment of qualitative model performance. While quantization helps mitigate memory overhead, it does not fully eliminate resource bottlenecks, especially for larger models. Our findings quantify the memory and energy constraints that must be considered for practical real-world deployments, offering concrete insights into the trade-offs between model size, inference performance, and efficiency. The exploration of LMs at the edge is still in its early stages. We hope this study provides a foundation for future research, guiding the refinement of models, the enhancement of inference efficiency, and the advancement of edge-centric AI systems.

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
* **Limits:** However, the inherent complexity of even these smaller models, combined with the limited computing resources of edge hardware, raises critical questions about the practical trade-offs in executing LM inference outside the cloud.
* **Signal Tags:** #ai

---


### Characterizing and Understanding Energy Footprint and Efficiency of Small Language Model on Edges
**Date:** 2025-11-18 | **Arxiv:** [2511.11624](https://hub.bitwiki.org/t/characterizing-and-understanding-energy-footprint-and-efficiency-of-small-language-model-on-edges/24251)

#### Abstract
Cloud-based large language models (LLMs) and their variants have significantly influenced real-world applications. Deploying smaller models (i.e., small language models (SLMs)) on edge devices offers additional advantages, such as reduced latency and independence from network connectivity. However, edge devices' limited computing resources and constrained energy budgets challenge efficient deployment. This study evaluates the power efficiency of five representative SLMs - Llama 3.2, Phi-3 Mini, TinyLlama, and Gemma 2 on Raspberry Pi 5, Jetson Nano, and Jetson Orin Nano (CPU and GPU configurations). Results show that Jetson Orin Nano with GPU acceleration achieves the highest energy-to-performance ratio, significantly outperforming CPU-based setups. Llama 3.2 provides the best balance of accuracy and power efficiency, while TinyLlama is well-suited for low-power environments at the cost of reduced accuracy. In contrast, Phi-3 Mini consumes the most energy despite its high accuracy. In addition, GPU acceleration, memory bandwidth, and model architecture are key in optimizing inference energy efficiency. Our empirical analysis offers practical insights for AI, smart systems, and mobile ad-hoc platforms to leverage tradeoffs from accuracy, inference latency, and power efficiency in energy-constrained environments.

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
* **Limits:** However, edge devices' limited computing resources and constrained energy budgets challenge efficient deployment.
* **Signal Tags:** #ai

---


### SLMQuant:Benchmarking Small Language Model Quantization for Practical Deployment
**Date:** 2025-11-18 | **Arxiv:** [2511.13023](https://hub.bitwiki.org/t/slmquant-benchmarking-small-language-model-quantization-for-practical-deployment/24180)

#### Abstract
Despite the growing interest in Small Language Models (SLMs) as resource-efficient alternatives to Large Language Models (LLMs), their deployment on edge devices remains challenging due to unresolved efficiency gaps in model compression. While quantization has proven effective for LLMs, its applicability to SLMs is significantly underexplored, with critical questions about differing quantization bottlenecks and efficiency profiles. This paper introduces SLMQuant, the first systematic benchmark for evaluating LLM compression techniques when applied to SLMs. Through comprehensive multi-track evaluations across diverse architectures and tasks, we analyze how state-of-the-art quantization methods perform on SLMs. Our findings reveal fundamental disparities between SLMs and LLMs in quantization sensitivity, demonstrating that direct transfer of LLM-optimized techniques leads to suboptimal results due to SLMs' unique architectural characteristics and training dynamics. We identify key factors governing effective SLM quantization and propose actionable design principles for SLM-tailored compression. SLMQuant establishes a foundational framework for advancing efficient SLM deployment on low-end devices in edge applications, and provides critical insights for deploying lightweight language models in resource-constrained scenarios.

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


### Enhancing Machine Learning Model Efficiency through Quantization and Bit Depth Optimization: A Performance Analysis on Healthcare Data
**Date:** 2025-11-18 | **Arxiv:** [2511.12568](https://hub.bitwiki.org/t/enhancing-machine-learning-model-efficiency-through-quantization-and-bit-depth-optimization-a-performance-analysis-on-healthcare-data/24131)

#### Abstract
This research aims to optimize intricate learning models by implementing quantization and bit-depth optimization techniques. The objective is to significantly cut time complexity while preserving model efficiency, thus addressing the challenge of extended execution times in intricate models. Two medical datasets were utilized as case studies to apply a Logistic Regression (LR) machine learning model. Using efficient quantization and bit depth optimization strategies the input data is downscaled from float64 to float32 and int32. The results demonstrated a significant reduction in time complexity, with only a minimal decrease in model accuracy post-optimization, showcasing the state-of-the-art optimization approach. This comprehensive study concludes that the impact of these optimization techniques varies depending on a set of parameters.

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


### EcoSpa: Efficient Transformer Training with Coupled Sparsity
**Date:** 2025-11-18 | **Arxiv:** [2511.11641](https://hub.bitwiki.org/t/ecospa-efficient-transformer-training-with-coupled-sparsity/23950)

#### Abstract
Transformers have become the backbone of modern AI, yet their high computational demands pose critical system challenges. While sparse training offers efficiency gains, existing methods fail to preserve critical structural relationships between weight matrices that interact multiplicatively in attention and feed-forward layers. This oversight leads to performance degradation at high sparsity levels. We introduce EcoSpa, an efficient structured sparse training method that jointly evaluates and sparsifies coupled weight matrix pairs, preserving their interaction patterns through aligned row/column removal. EcoSpa introduces a new granularity for calibrating structural component importance and performs coupled estimation and sparsification across both pre-training and fine-tuning scenarios. Evaluations demonstrate substantial improvements: EcoSpa enables efficient training of LLaMA-1B with 50\% memory reduction and 21\% faster training, achieves $2.2\times$ model compression on GPT-2-Medium with $2.4$ lower perplexity, and delivers $1.6\times$ inference speedup. The approach uses standard PyTorch operations, requiring no custom hardware or kernels, making efficient transformer training accessible on commodity hardware.

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


### Choose Your Model Size: Any Compression of Large Language Models Without Re-Computation
**Date:** 2025-11-11 | **Arxiv:** [2502.01717](https://hub.bitwiki.org/t/choose-your-model-size-any-compression-of-large-language-models-without-re-computation/22837)

#### Abstract
The adoption of Foundation Models in resource-constrained environments remains challenging due to their large size and inference costs. A promising way to overcome these limitations is post-training compression, which aims to balance reduced model size against performance degradation. This work presents Any Compression via Iterative Pruning (ACIP), a novel algorithmic approach to determine a compression-performance trade-off from a single stochastic gradient descent run. To achieve parameter efficiency, we use an SVD-reparametrization of linear layers and iteratively prune their singular values with a sparsity-inducing penalty. Importantly, the pruning order of the parameters is used to derive a global score map that allows compressing a model to any target size without re-computation. We evaluate ACIP on a large selection of open-weight LLMs and downstream tasks, demonstrating state-of-the-art results compared to existing factorization-based compression methods. We also show that ACIP seamlessly complements common quantization-based compression techniques.

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


### TT-Prune: Joint Model Pruning and Resource Allocation for Communication-efficient Time-triggered Federated Learning
**Date:** 2025-11-07 | **Arxiv:** [2511.04653](https://hub.bitwiki.org/t/tt-prune-joint-model-pruning-and-resource-allocation-for-communication-efficient-time-triggered-federated-learning/22082)

#### Abstract
Federated learning (FL) offers new opportunities in machine learning, particularly in addressing data privacy concerns. In contrast to conventional event-based federated learning, time-triggered federated learning (TT-Fed), as a general form of both asynchronous and synchronous FL, clusters users into different tiers based on fixed time intervals. However, the FL network consists of a growing number of user devices with limited wireless bandwidth, consequently magnifying issues such as stragglers and communication overhead. In this paper, we introduce adaptive model pruning to wireless TT-Fed systems and study the problem of jointly optimizing the pruning ratio and bandwidth allocation to minimize the training loss while ensuring minimal learning latency. To answer this question, we perform convergence analysis on the gradient l_2 norm of the TT-Fed model based on model pruning. Based on the obtained convergence upper bound, a joint optimization problem of pruning ratio and wireless bandwidth is formulated to minimize the model training loss under a given delay threshold. Then, we derive closed-form solutions for wireless bandwidth and pruning ratio using Karush-Kuhn-Tucker(KKT) conditions. The simulation results show that model pruning could reduce the communication cost by 40% while maintaining the model performance at the same level.

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
* **Limits:** However, the FL network consists of a growing number of user devices with limited wireless bandwidth, consequently magnifying issues such as stragglers and communication overhead.
* **Signal Tags:** #ai

---


### E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products
**Date:** 2025-11-04 | **Arxiv:** [2501.19216](https://hub.bitwiki.org/t/e2former-an-efficient-and-equivariant-transformer-with-linear-scaling-tensor-products/21382)

#### Abstract
Equivariant Graph Neural Networks (EGNNs) have demonstrated significant success in modeling microscale systems, including those in chemistry, biology and materials science. However, EGNNs face substantial computational challenges due to the high cost of constructing edge features via spherical tensor products, making them impractical for large-scale systems. To address this limitation, we introduce E2Former, an equivariant and efficient transformer architecture that incorporates the Wigner $6j$ convolution (Wigner $6j$ Conv). By shifting the computational burden from edges to nodes, the Wigner $6j$ Conv reduces the complexity from $O(|\mathcal{E}|)$ to $ O(| \mathcal{V}|)$ while preserving both the model's expressive power and rotational equivariance. We show that this approach achieves a 7x-30x speedup compared to conventional $\mathrm{SO}(3)$ convolutions. Furthermore, our empirical results demonstrate that the derived E2Former mitigates the computational challenges of existing approaches without compromising the ability to capture detailed geometric information. This development could suggest a promising direction for scalable and efficient molecular modeling.

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
* **Limits:** However, EGNNs face substantial computational challenges due to the high cost of constructing edge features via spherical tensor products, making them impractical for large-scale systems.
* **Signal Tags:** #ai

---


### Sparse Model Inversion: Efficient Inversion of Vision Transformers for Data-Free Applications
**Date:** 2025-11-03 | **Arxiv:** [2510.27186](https://hub.bitwiki.org/t/sparse-model-inversion-efficient-inversion-of-vision-transformers-for-data-free-applications/20904)

#### Abstract
Model inversion, which aims to reconstruct the original training data from pre-trained discriminative models, is especially useful when the original training data is unavailable due to privacy, usage rights, or size constraints. However, existing dense inversion methods attempt to reconstruct the entire image area, making them extremely inefficient when inverting high-resolution images from large-scale Vision Transformers (ViTs). We further identify two underlying causes of this inefficiency: the redundant inversion of noisy backgrounds and the unintended inversion of spurious correlations--a phenomenon we term "hallucination" in model inversion. To address these limitations, we propose a novel sparse model inversion strategy, as a plug-and-play extension to speed up existing dense inversion methods with no need for modifying their original loss functions. Specifically, we selectively invert semantic foregrounds while stopping the inversion of noisy backgrounds and potential spurious correlations. Through both theoretical and empirical studies, we validate the efficacy of our approach in achieving significant inversion acceleration (up to 3.79 faster) while maintaining comparable or even enhanced downstream performance in data-free model quantization and data-free knowledge transfer. Code is available at https://github.com/Egg-Hu/SMI.

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
* **Limits:** However, existing dense inversion methods attempt to reconstruct the entire image area, making them extremely inefficient when inverting high-resolution images from large-scale Vision Transformers (ViTs).
* **Signal Tags:** #ai

---


### Provable Sample-Efficient Transfer Learning Conditional Diffusion Models via Representation Learning
**Date:** 2025-10-28 | **Arxiv:** [2502.04491](https://hub.bitwiki.org/t/provable-sample-efficient-transfer-learning-conditional-diffusion-models-via-representation-learning/19885)

#### Abstract
While conditional diffusion models have achieved remarkable success in various applications, they require abundant data to train from scratch, which is often infeasible in practice. To address this issue, transfer learning has emerged as an essential paradigm in small data regimes. Despite its empirical success, the theoretical underpinnings of transfer learning conditional diffusion models remain unexplored. In this paper, we take the first step towards understanding the sample efficiency of transfer learning conditional diffusion models through the lens of representation learning. Inspired by practical training procedures, we assume that there exists a low-dimensional representation of conditions shared across all tasks. Our analysis shows that with a well-learned representation from source tasks, the samplecomplexity of target tasks can be reduced substantially. In addition, we investigate the practical implications of our theoretical results in several real-world applications of conditional diffusion models. Numerical experiments are also conducted to verify our results.

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


### Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders
**Date:** 2025-10-28 | **Arxiv:** [2510.22332](https://hub.bitwiki.org/t/transformer-key-value-memories-are-nearly-as-interpretable-as-sparse-autoencoders/19825)

#### Abstract
Recent interpretability work on large language models (LLMs) has been increasingly dominated by a feature-discovery approach with the help of proxy modules. Then, the quality of features learned by, e.g., sparse auto-encoders (SAEs), is evaluated. This paradigm naturally raises a critical question: do such learned features have better properties than those already represented within the original model parameters, and unfortunately, only a few studies have made such comparisons systematically so far. In this work, we revisit the interpretability of feature vectors stored in feed-forward (FF) layers, given the perspective of FF as key-value memories, with modern interpretability benchmarks. Our extensive evaluation revealed that SAE and FFs exhibits a similar range of interpretability, although SAEs displayed an observable but minimal improvement in some aspects. Furthermore, in certain aspects, surprisingly, even vanilla FFs yielded better interpretability than the SAEs, and features discovered in SAEs and FFs diverged. These bring questions about the advantage of SAEs from both perspectives of feature quality and faithfulness, compared to directly interpreting FF feature vectors, and FF key-value parameters serve as a strong baseline in modern interpretability research.

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


### Self-Refining Language Model Anonymizers via Adversarial Distillation
**Date:** 2025-10-27 | **Arxiv:** [2506.01420](https://hub.bitwiki.org/t/self-refining-language-model-anonymizers-via-adversarial-distillation/19741)

#### Abstract
Large language models (LLMs) are increasingly used in sensitive domains, where their ability to infer personal data from seemingly benign text introduces emerging privacy risks. While recent LLM-based anonymization methods help mitigate such risks, they often rely on proprietary models (e.g., GPT-4), raising concerns about cost and the potential exposure of sensitive data to untrusted external systems. To address this, we introduce SElf-refining Anonymization with Language model (SEAL), a novel distillation framework for training small language models (SLMs) to perform effective anonymization without relying on external models at inference time. SEAL leverages adversarial interactions between an LLM anonymizer and an inference model to collect trajectories of anonymized texts and inferred attributes, which are then used to distill anonymization and critique capabilities into SLMs through supervised fine-tuning and preference learning. The resulting models learn both to anonymize text and to evaluate their outputs, enabling iterative improvement of anonymization quality via self-refinement. Experiments on SynthPAI, a dataset of synthetic personal profiles and text comments, demonstrate that SLMs trained with SEAL achieve substantial improvements in anonymization capabilities. Notably, 8B models attain a privacy-utility trade-off comparable to that of the GPT-4 anonymizer and, with self-refinement, even surpass it in terms of privacy protection. These results highlight the effectiveness of our adversarial distillation framework for training SLMs as efficient anonymizers.

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


### ARA: Adaptive Rank Allocation for Efficient Large Language Model SVD Compression
**Date:** 2025-10-23 | **Arxiv:** [2510.19389](https://hub.bitwiki.org/t/ara-adaptive-rank-allocation-for-efficient-large-language-model-svd-compression/18885)

#### Abstract
In the field of large language model (LLM) compression, singular value decomposition (SVD) is a widely studied and adopted low-rank decomposition technique. Since SVD operates exclusively on linear modules, and these modules in LLMs are separated by nonlinear components, SVD can only be applied independently to each linear module. Under a global compression ratio constraint, determining the appropriate rank for different linear modules becomes a critical problem. Existing approaches, such as heuristic algorithms and mask-based training, have made progress in addressing this challenge. However, these methods still suffer from several limitations: heuristic algorithms explore the solution space within restricted regions, while mask-based training struggles to efficiently capture the relationship between singular value spectra and trainable parameters. More importantly, current methods overlook the key property that the gain function is non-smooth at a compression ratio of 1, which often leads the training process to suboptimal local minima. To address these issues, we propose an Adaptive Rank Allocation (ARA) method. Specifically, (1) ARA introduces a dedicated mask design that enables efficient mapping and updating between retained ranks and trainable parameters; and (2) it employs an additional loss function to guide parameter selection toward globally optimal solutions. Experimental results demonstrate that ARA achieves state-of-the-art performance. On the LLaMA2-7B model with a 80\% compression ratio, ARA reduces perplexity on WikiText2 from 8.38 to 6.42 and improves average zero-shot task accuracy by 9.72 percentage points compared with uniform compression. These results highlight the effectiveness of our method for rank allocation in SVD-based LLM compression.

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
* **Limits:** However, these methods still suffer from several limitations: heuristic algorithms explore the solution space within restricted regions, while mask-based training struggles to efficiently capture the relationship between singular value spectra and trainable parameters.
* **Signal Tags:** #ai

---


### AMAuT: A Flexible and Efficient Multiview Audio Transformer Framework Trained from Scratch
**Date:** 2025-10-23 | **Arxiv:** [2510.19368](https://hub.bitwiki.org/t/amaut-a-flexible-and-efficient-multiview-audio-transformer-framework-trained-from-scratch/18950)

#### Abstract
Recent foundational models, SSAST, EAT, HuBERT, Qwen-Audio, and Audio Flamingo, achieve top-tier results across standard audio benchmarks but are limited by fixed input rates and durations, hindering their reusability. This paper introduces the Augmentation-driven Multiview Audio Transformer (AMAuT), a training-from-scratch framework that eliminates the dependency on pre-trained weights while supporting arbitrary sample rates and audio lengths. AMAuT integrates four key components: (1) augmentation-driven multiview learning for robustness, (2) a conv1 + conv7 + conv1 one-dimensional CNN bottleneck for stable temporal encoding, (3) dual CLS + TAL tokens for bidirectional context representation, and (4) test-time adaptation/augmentation (TTA^2) to improve inference reliability. Experiments on five public benchmarks, AudioMNIST, SpeechCommands V1 & V2, VocalSound, and CochlScene, show that AMAuT achieves accuracies up to 99.8% while consuming less than 3% of the GPU hours required by comparable pre-trained models. Thus, AMAuT presents a highly efficient and flexible alternative to large pre-trained models, making state-of-the-art audio classification accessible in computationally constrained settings.

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


### Transformer Redesign for Late Fusion of Audio-Text Features on Ultra-Low-Power Edge Hardware
**Date:** 2025-10-22 | **Arxiv:** [2510.18036](https://hub.bitwiki.org/t/transformer-redesign-for-late-fusion-of-audio-text-features-on-ultra-low-power-edge-hardware/18673)

#### Abstract
Deploying emotion recognition systems in real-world environments where devices must be small, low-power, and private remains a significant challenge. This is especially relevant for applications such as tension monitoring, conflict de-escalation, and responsive wearables, where cloud-based solutions are impractical. Multimodal emotion recognition has advanced through deep learning, but most systems remain unsuitable for deployment on ultra-constrained edge devices. Prior work typically relies on powerful hardware, lacks real-time performance, or uses unimodal input. This paper addresses that gap by presenting a hardware-aware emotion recognition system that combines acoustic and linguistic features using a late-fusion architecture optimised for Edge TPU. The design integrates a quantised transformer-based acoustic model with frozen keyword embeddings from a DSResNet-SE network, enabling real-time inference within a 1.8MB memory budget and 21-23ms latency. The pipeline ensures spectrogram alignment between training and deployment using MicroFrontend and MLTK. Evaluation on re-recorded, segmented IEMOCAP samples captured through the Coral Dev Board Micro microphone shows a 6.3% macro F1 improvement over unimodal baselines. This work demonstrates that accurate, real-time multimodal emotion inference is achievable on microcontroller-class edge platforms through task-specific fusion and hardware-guided model design.

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


### Efficient Long-context Language Model Training by Core Attention Disaggregation
**Date:** 2025-10-22 | **Arxiv:** [2510.18121](https://hub.bitwiki.org/t/efficient-long-context-language-model-training-by-core-attention-disaggregation/18572)

#### Abstract
We present core attention disaggregation (CAD), a technique that improves long-context large language model training by decoupling the core attention computation, softmax(QK^T)V, from the rest of the model and executing it on a separate pool of devices. In existing systems, core attention is colocated with other layers; at long context lengths, its quadratic compute growth compared to the near-linear growth of other components causes load imbalance and stragglers across data and pipeline parallel groups. CAD is enabled by two observations. First, core attention is stateless: it has no trainable parameters and only minimal transient data, so balancing reduces to scheduling compute-bound tasks. Second, it is composable: modern attention kernels retain high efficiency when processing fused batches of token-level shards with arbitrary lengths. CAD partitions core attention into token-level tasks and dispatches them to dedicated attention servers, which dynamically rebatch tasks to equalize compute without sacrificing kernel efficiency. We implement CAD in a system called DistCA, which uses a ping-pong execution scheme to fully overlap communication with computation and in-place execution on attention servers to reduce memory use. On 512 H200 GPUs and context lengths up to 512k tokens, DistCA improves end-to-end training throughput by up to 1.35x, eliminates data and pipeline parallel stragglers, and achieves near-perfect compute and memory balance.

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


### Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model
**Date:** 2025-10-22 | **Arxiv:** [2510.18165](https://hub.bitwiki.org/t/saber-an-efficient-sampling-with-adaptive-acceleration-and-backtracking-enhanced-remasking-for-diffusion-language-model/18680)

#### Abstract
Diffusion language models (DLMs) are emerging as a powerful and promising alternative to the dominant autoregressive paradigm, offering inherent advantages in parallel generation and bidirectional context modeling. However, the performance of DLMs on code generation tasks, which have stronger structural constraints, is significantly hampered by the critical trade-off between inference speed and output quality. We observed that accelerating the code generation process by reducing the number of sampling steps usually leads to a catastrophic collapse in performance. In this paper, we introduce efficient Sampling with Adaptive acceleration and Backtracking Enhanced Remasking (i.e., Saber), a novel training-free sampling algorithm for DLMs to achieve better inference speed and output quality in code generation. Specifically, Saber is motivated by two key insights in the DLM generation process: 1) it can be adaptively accelerated as more of the code context is established; 2) it requires a backtracking mechanism to reverse the generated tokens. Extensive experiments on multiple mainstream code generation benchmarks show that Saber boosts Pass@1 accuracy by an average improvement of 1.9% over mainstream DLM sampling methods, meanwhile achieving an average 251.4% inference speedup. By leveraging the inherent advantages of DLMs, our work significantly narrows the performance gap with autoregressive models in code generation.

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
* **Limits:** However, the performance of DLMs on code generation tasks, which have stronger structural constraints, is significantly hampered by the critical trade-off between inference speed and output quality.
* **Signal Tags:** #ai

---


### Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning
**Date:** 2025-10-21 | **Arxiv:** [2510.16877](https://hub.bitwiki.org/t/fly-cl-a-fly-inspired-framework-for-enhancing-efficient-decorrelation-and-reduced-training-time-in-pre-trained-model-based-continual-representation-learning/18139)

#### Abstract
Using a nearly-frozen pretrained model, the continual representation learning paradigm reframes parameter updates as a similarity-matching problem to mitigate catastrophic forgetting. However, directly leveraging pretrained features for downstream tasks often suffers from multicollinearity in the similarity-matching stage, and more advanced methods can be computationally prohibitive for real-time, low-latency applications. Inspired by the fly olfactory circuit, we propose Fly-CL, a bio-inspired framework compatible with a wide range of pretrained backbones. Fly-CL substantially reduces training time while achieving performance comparable to or exceeding that of current state-of-the-art methods. We theoretically show how Fly-CL progressively resolves multicollinearity, enabling more effective similarity matching with low time complexity. Extensive simulation experiments across diverse network architectures and data regimes validate Fly-CL's effectiveness in addressing this challenge through a biologically inspired design. Code is available at https://github.com/gfyddha/Fly-CL.

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
* **Limits:** However, directly leveraging pretrained features for downstream tasks often suffers from multicollinearity in the similarity-matching stage, and more advanced methods can be computationally prohibitive for real-time, low-latency applications.
* **Signal Tags:** #ai

---


### Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey
**Date:** 2025-10-21 | **Arxiv:** [2510.17111](https://hub.bitwiki.org/t/efficient-vision-language-action-models-for-embodied-manipulation-a-systematic-survey/18317)

#### Abstract
Vision-Language-Action (VLA) models extend vision-language models to embodied control by mapping natural-language instructions and visual observations to robot actions. Despite their capabilities, VLA systems face significant challenges due to their massive computational and memory demands, which conflict with the constraints of edge platforms such as on-board mobile manipulators that require real-time performance. Addressing this tension has become a central focus of recent research. In light of the growing efforts toward more efficient and scalable VLA systems, this survey provides a systematic review of approaches for improving VLA efficiency, with an emphasis on reducing latency, memory footprint, and training and inference costs. We categorize existing solutions into four dimensions: model architecture, perception feature, action generation, and training/inference strategies, summarizing representative techniques within each category. Finally, we discuss future trends and open challenges, highlighting directions for advancing efficient embodied intelligence.

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


### Sparse Transformer Architectures via Regularized Wasserstein Proximal Operator with $L_1$ Prior
**Date:** 2025-10-21 | **Arxiv:** [2510.16356](https://hub.bitwiki.org/t/sparse-transformer-architectures-via-regularized-wasserstein-proximal-operator-with-l-1-prior/18013)

#### Abstract
In this work, we propose a sparse transformer architecture that incorporates prior information about the underlying data distribution directly into the transformer structure of the neural network. The design of the model is motivated by a special optimal transport problem, namely the regularized Wasserstein proximal operator, which admits a closed-form solution and turns out to be a special representation of transformer architectures. Compared with classical flow-based models, the proposed approach improves the convexity properties of the optimization problem and promotes sparsity in the generated samples. Through both theoretical analysis and numerical experiments, including applications in generative modeling and Bayesian inverse problems, we demonstrate that the sparse transformer achieves higher accuracy and faster convergence to the target distribution than classical neural ODE-based methods.

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


### SOLE: Hardware-Software Co-design of Softmax and LayerNorm for Efficient Transformer Inference
**Date:** 2025-10-21 | **Arxiv:** [2510.17189](https://hub.bitwiki.org/t/sole-hardware-software-co-design-of-softmax-and-layernorm-for-efficient-transformer-inference/18176)

#### Abstract
Transformers have shown remarkable performance in both natural language processing (NLP) and computer vision (CV) tasks. However, their real-time inference speed and efficiency are limited due to the inefficiency in Softmax and Layer Normalization (LayerNorm). Previous works based on function approximation suffer from inefficient implementation as they place emphasis on computation while disregarding memory overhead concerns. Moreover, such methods rely on retraining to compensate for approximation error which can be costly and inconvenient.   In this paper, we present SOLE, a hardware-software co-design for Softmax and LayerNorm which is composed of E2Softmax and AILayerNorm. E2Softmax utilizes log2 quantization of exponent function and log-based division to approximate Softmax while AILayerNorm adopts low-precision statistic calculation. Compared with state-of-the-art designs, we achieve both low-precision calculation and low bit-width storage on Softmax and LayerNorm. Experiments show that SOLE maintains inference accuracy without retraining while offering orders of magnitude speedup and energy savings over GPU, achieving 3.04x, 3.86x energy-efficiency improvements and 2.82x, 3.32x area-efficiency improvements over prior state-of-the-art custom hardware for Softmax and LayerNorm, respectively.

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
* **Layer:** Hardware
* **Limits:** However, their real-time inference speed and efficiency are limited due to the inefficiency in Softmax and Layer Normalization (LayerNorm).
* **Signal Tags:** #ai

---


### MX+: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving
**Date:** 2025-10-17 | **Arxiv:** [2510.14557](https://hub.bitwiki.org/t/mx-pushing-the-limits-of-microscaling-formats-for-efficient-large-language-model-serving/17495)

#### Abstract
Reduced-precision data formats are crucial for cost-effective serving of large language models (LLMs). While numerous reduced-precision formats have been introduced thus far, they often require intrusive modifications to the software frameworks or are rather unconventional for widespread adoption across hardware vendors. In this paper, we instead focus on recent industry-driven variants of block floating-point (BFP) formats and conduct a comprehensive analysis to push their limits for efficient LLM serving. Our analysis shows that existing ultra low-bit BFP variants struggle to provide reasonable language model performance due to outlier values in blocks. To address the outliers with BFPs, we propose MX+, a cost-effective and non-intrusive extension designed for seamless integration into the microscaling (MX) formats. MX+ builds on the key insight that the outlier does not need to use its exponent field in the element data type, which allows us to repurpose the exponent field as an extended mantissa to increase the precision of the outlier element. Our evaluation shows that MX+ achieves significantly higher model performance compared to the 4-bit MX format (MXFP4) with negligible storage overhead and slowdown, thus offering a compelling alternative to MXFP4 or MXFP6 for efficient LLM inference.

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


### Efficient Autoregressive Inference for Transformer Probabilistic Models
**Date:** 2025-10-13 | **Arxiv:** [2510.09477](https://hub.bitwiki.org/t/efficient-autoregressive-inference-for-transformer-probabilistic-models/16222)

#### Abstract
Transformer-based models for amortized probabilistic inference, such as neural processes, prior-fitted networks, and tabular foundation models, excel at single-pass marginal prediction. However, many real-world applications, from signal interpolation to multi-column tabular predictions, require coherent joint distributions that capture dependencies between predictions. While purely autoregressive architectures efficiently generate such distributions, they sacrifice the flexible set-conditioning that makes these models powerful for meta-learning. Conversely, the standard approach to obtain joint distributions from set-based models requires expensive re-encoding of the entire augmented conditioning set at each autoregressive step. We introduce a causal autoregressive buffer that preserves the advantages of both paradigms. Our approach decouples context encoding from updating the conditioning set. The model processes the context once and caches it. A dynamic buffer then captures target dependencies: as targets are incorporated, they enter the buffer and attend to both the cached context and previously buffered targets. This enables efficient batched autoregressive generation and one-pass joint log-likelihood evaluation. A unified training strategy allows seamless integration of set-based and autoregressive modes at minimal additional cost. Across synthetic functions, EEG signals, cognitive models, and tabular data, our method matches predictive accuracy of strong baselines while delivering up to 20 times faster joint sampling. Our approach combines the efficiency of autoregressive generative models with the representational power of set-based conditioning, making joint prediction practical for transformer-based probabilistic models.

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
* **Limits:** However, many real-world applications, from signal interpolation to multi-column tabular predictions, require coherent joint distributions that capture dependencies between predictions.
* **Signal Tags:** #ai

---


### Recycling Pretrained Checkpoints: Orthogonal Growth of Mixture-of-Experts for Efficient Large Language Model Pre-Training
**Date:** 2025-10-10 | **Arxiv:** [2510.08008](https://hub.bitwiki.org/t/recycling-pretrained-checkpoints-orthogonal-growth-of-mixture-of-experts-for-efficient-large-language-model-pre-training/15968)

#### Abstract
The rapidly increasing computational cost of pretraining Large Language Models necessitates more efficient approaches. Numerous computational costs have been invested in existing well-trained checkpoints, but many of them remain underutilized due to engineering constraints or limited model capacity. To efficiently reuse this "sunk" cost, we propose to recycle pretrained checkpoints by expanding their parameter counts and continuing training. We propose orthogonal growth method well-suited for converged Mixture-of-Experts model: interpositional layer copying for depth growth and expert duplication with injected noise for width growth. To determine the optimal timing for such growth across checkpoints sequences, we perform comprehensive scaling experiments revealing that the final accuracy has a strong positive correlation with the amount of sunk cost, indicating that greater prior investment leads to better performance. We scale our approach to models with 70B parameters and over 1T training tokens, achieving 10.66% accuracy gain over training from scratch under the same additional compute budget. Our checkpoint recycling approach establishes a foundation for economically efficient large language model pretraining.

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


### Rethinking Decoders for Transformer-based Semantic Segmentation: A Compression Perspective
**Date:** 2025-10-10 | **Arxiv:** [2411.03033](https://hub.bitwiki.org/t/rethinking-decoders-for-transformer-based-semantic-segmentation-a-compression-perspective/16137)

#### Abstract
State-of-the-art methods for Transformer-based semantic segmentation typically adopt Transformer decoders that are used to extract additional embeddings from image embeddings via cross-attention, refine either or both types of embeddings via self-attention, and project image embeddings onto the additional embeddings via dot-product. Despite their remarkable success, these empirical designs still lack theoretical justifications or interpretations, thus hindering potentially principled improvements. In this paper, we argue that there are fundamental connections between semantic segmentation and compression, especially between the Transformer decoders and Principal Component Analysis (PCA). From such a perspective, we derive a white-box, fully attentional DEcoder for PrIncipled semantiC segemenTation (DEPICT), with the interpretations as follows: 1) the self-attention operator refines image embeddings to construct an ideal principal subspace that aligns with the supervision and retains most information; 2) the cross-attention operator seeks to find a low-rank approximation of the refined image embeddings, which is expected to be a set of orthonormal bases of the principal subspace and corresponds to the predefined classes; 3) the dot-product operation yields compact representation for image embeddings as segmentation masks. Experiments conducted on dataset ADE20K find that DEPICT consistently outperforms its black-box counterpart, Segmenter, and it is light weight and more robust.

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


### ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization
**Date:** 2025-10-07 | **Arxiv:** [2505.02819](https://hub.bitwiki.org/t/replaceme-network-simplification-via-depth-pruning-and-transformer-block-linearization/15005)

#### Abstract
We introduce ReplaceMe, a generalized training-free depth pruning method that effectively replaces transformer blocks with a linear operation, while maintaining high performance for low compression ratios. In contrast to conventional pruning approaches that require additional training or fine-tuning, our approach requires only a small calibration dataset that is used to estimate a linear transformation, which approximates the pruned blocks. The estimated linear mapping can be seamlessly merged with the remaining transformer blocks, eliminating the need for any additional network parameters. Our experiments show that ReplaceMe consistently outperforms other training-free approaches and remains highly competitive with state-of-the-art pruning methods that involve extensive retraining/fine-tuning and architectural modifications. Applied to several large language models (LLMs), ReplaceMe achieves up to 25% pruning while retaining approximately 90% of the original model's performance on open benchmarks - without any training or healing steps, resulting in minimal computational overhead (see Fig.1). We provide an open-source library implementing ReplaceMe alongside several state-of-the-art depth pruning techniques, available at https://github.com/mts-ai/ReplaceMe.

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


### Pixels Versus Priors: Controlling Knowledge Priors in Vision-Language Models through Visual Counterfacts
**Date:** 2025-09-30 | **Arxiv:** [2505.17127](https://hub.bitwiki.org/t/pixels-versus-priors-controlling-knowledge-priors-in-vision-language-models-through-visual-counterfacts/13158)

#### Abstract
Multimodal Large Language Models (MLLMs) perform well on tasks such as visual question answering, but it remains unclear whether their reasoning relies more on memorized world knowledge or on the visual information present in the input image. To investigate this, we introduce Visual CounterFact, a new dataset of visually-realistic counterfactuals that put world knowledge priors (e.g, red strawberry) into direct conflict with visual input (e.g, blue strawberry). Using Visual CounterFact, we show that model predictions initially reflect memorized priors, but shift toward visual evidence in mid-to-late layers. This dynamic reveals a competition between the two modalities, with visual input ultimately overriding priors during evaluation. To control this behavior, we propose Pixels Versus Priors (PvP) steering vectors, a mechanism for controlling model outputs toward either world knowledge or visual input through activation-level interventions. On average, PvP successfully shifts 99.3% of color and 80.8% of size predictions from priors to counterfactuals. Together, these findings offer new tools for interpreting and controlling factual behavior in multimodal models.

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


### BiHDTrans: binary hyperdimensional transformer for efficient multivariate time series classification
**Date:** 2025-09-30 | **Arxiv:** [2509.24425](https://hub.bitwiki.org/t/bihdtrans-binary-hyperdimensional-transformer-for-efficient-multivariate-time-series-classification/12688)

#### Abstract
The proliferation of Internet-of-Things (IoT) devices has led to an unprecedented volume of multivariate time series (MTS) data, requiring efficient and accurate processing for timely decision-making in resource-constrained edge environments. Hyperdimensional (HD) computing, with its inherent efficiency and parallelizability, has shown promise in classification tasks but struggles to capture complex temporal patterns, while Transformers excel at sequence modeling but incur high computational and memory overhead. We introduce BiHDTrans, an efficient neurosymbolic binary hyperdimensional Transformer that integrates self-attention into the HD computing paradigm, unifying the representational efficiency of HD computing with the temporal modeling power of Transformers. Empirically, BiHDTrans outperforms state-of-the-art (SOTA) HD computing models by at least 14.47% and achieves 6.67% higher accuracy on average than SOTA binary Transformers. With hardware acceleration on FPGA, our pipelined implementation leverages the independent and identically distributed properties of high-dimensional representations, delivering 39.4 times lower inference latency than SOTA binary Transformers. Theoretical analysis shows that binarizing in holographic high-dimensional space incurs significantly less information distortion than directly binarizing neural networks, explaining BiHDTrans's superior accuracy. Furthermore, dimensionality experiments confirm that BiHDTrans remains competitive even with a 64% reduction in hyperspace dimensionality, surpassing SOTA binary Transformers by 1-2% in accuracy with 4.4 times less model size, as well as further reducing the latency by 49.8% compare to the full-dimensional baseline. Together, these contributions bridge the gap between the expressiveness of Transformers and the efficiency of HD computing, enabling accurate, scalable, and low-latency MTS classification.

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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### A GREAT Architecture for Edge-Based Graph Problems Like TSP
**Date:** 2025-09-30 | **Arxiv:** [2408.16717](https://hub.bitwiki.org/t/a-great-architecture-for-edge-based-graph-problems-like-tsp/12965)

#### Abstract
In the last years, an increasing number of learning-based approaches have been proposed to tackle combinatorial optimization problems such as routing problems. Many of these approaches are based on graph neural networks (GNNs) or related transformers, operating on the Euclidean coordinates representing the routing problems. However, such models are ill-suited for a wide range of real-world problems that feature non-Euclidean and asymmetric edge costs. To overcome this limitation, we propose a novel GNN-based and edge-focused neural model called Graph Edge Attention Network (GREAT). Using GREAT as an encoder to capture the properties of a routing problem instance, we build a reinforcement learning framework which we apply to both Euclidean and non-Euclidean variants of vehicle routing problems such as Traveling Salesman Problem, Capacitated Vehicle Routing Problem and Orienteering Problem. Our framework is among the first to tackle non-Euclidean variants of these problems and achieves competitive results among learning-based benchmarks.

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
* **Limits:** However, such models are ill-suited for a wide range of real-world problems that feature non-Euclidean and asymmetric edge costs.
* **Signal Tags:** #ai

---


### RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours
**Date:** 2025-09-30 | **Arxiv:** [2505.10271](https://hub.bitwiki.org/t/rainpro-8-an-efficient-deep-learning-model-to-estimate-rainfall-probabilities-over-8-hours/13003)

#### Abstract
We present a deep learning model for high-resolution probabilistic precipitation forecasting over an 8-hour horizon in Europe, overcoming the limitations of radar-only deep learning models with short forecast lead times. Our model efficiently integrates multiple data sources - including radar, satellite, and physics-based numerical weather prediction (NWP) - while capturing long-range interactions, resulting in accurate forecasts with robust uncertainty quantification through consistent probabilistic maps. Featuring a compact architecture, it enables more efficient training and faster inference than existing models. Extensive experiments demonstrate that our model surpasses current operational NWP systems, extrapolation-based methods, and deep-learning nowcasting models, setting a new standard for high-resolution precipitation forecasting in Europe, ensuring a balance between accuracy, interpretability, and computational efficiency.

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


### ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models
**Date:** 2025-09-29 | **Arxiv:** [2509.21991](https://hub.bitwiki.org/t/ergo-efficient-high-resolution-visual-understanding-for-vision-language-models/12192)

#### Abstract
Efficient processing of high-resolution images is crucial for real-world vision-language applications. However, existing Large Vision-Language Models (LVLMs) incur substantial computational overhead due to the large number of vision tokens. With the advent of "thinking with images" models, reasoning now extends beyond text to the visual domain. This capability motivates our two-stage "coarse-to-fine" reasoning pipeline: first, a downsampled image is analyzed to identify task-relevant regions; then, only these regions are cropped at full resolution and processed in a subsequent reasoning stage. This approach reduces computational cost while preserving fine-grained visual details where necessary. A major challenge lies in inferring which regions are truly relevant to a given query. Recent related methods often fail in the first stage after input-image downsampling, due to perception-driven reasoning, where clear visual information is required for effective reasoning. To address this issue, we propose ERGO (Efficient Reasoning & Guided Observation) that performs reasoning-driven perception-leveraging multimodal context to determine where to focus. Our model can account for perceptual uncertainty, expanding the cropped region to cover visually ambiguous areas for answering questions. To this end, we develop simple yet effective reward components in a reinforcement learning framework for coarse-to-fine perception. Across multiple datasets, our approach delivers higher accuracy than the original model and competitive methods, with greater efficiency. For instance, ERGO surpasses Qwen2.5-VL-7B on the V* benchmark by 4.7 points while using only 23% of the vision tokens, achieving a 3x inference speedup. The code and models can be found at: https://github.com/nota-github/ERGO.

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
* **Limits:** However, existing Large Vision-Language Models (LVLMs) incur substantial computational overhead due to the large number of vision tokens.
* **Signal Tags:** #ai

---


### Holographic Knowledge Manifolds: A Novel Pipeline for Continual Learning Without Catastrophic Forgetting in Large Language Models
**Date:** 2025-09-16 | **Arxiv:** [2509.10518](https://hub.bitwiki.org/t/holographic-knowledge-manifolds-a-novel-pipeline-for-continual-learning-without-catastrophic-forgetting-in-large-language-models/9405)

#### Abstract
We introduce the Holographic Knowledge Manifold (HKM), a four-phase pipeline that achieves zero catastrophic forgetting in AI knowledge representation while maintaining minimal memory growth and high efficiency. Leveraging fractal quantization, probabilistic entanglement, and dynamic diffraction chipping, HKM compresses knowledge substrates by 3x with 67% storage savings, integrates holographically at 100%, and supports over 1,020 updates with 1% growth per increment. In experiments on combined WikiText and FB15k datasets (scaled to 2,997 nodes), we demonstrate industry-leading performance: 0% forgetting (infinite improvement over GEM baselines), 3x compression, and 53% training time reduction on consumer GPU hardware. Hypothetical cost analyses project $92.4M savings over 5 years at petabyte scale, with 21.2% energy reduction and 33% lower carbon footprint. This work hypothesizes a paradigm shift for public large language models (LLMs), enabling "eternal" adaptation without retraining. Future extensions to multimodal fusion and quantum hardware could further democratize scalable AI, potentially reducing fine-tuning costs by 60-80% for models like Llama-3 or Grok-4. Code, datasets, and full results are publicly available for reproducibility.

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
