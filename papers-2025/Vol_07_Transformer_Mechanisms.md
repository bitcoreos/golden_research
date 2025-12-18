# Vol 07 Transformer Mechanisms
*Enriched by BITCOREOS | Phase 4 Batch 2*

---

### Layer Importance for Mathematical Reasoning is Forged in Pre-Training and Invariant after Post-Training
**Date:** 2025-11-06 | **Arxiv:** [2506.22638](https://arxiv.org/abs/2506.22638)

#### Abstract
Large language models improve at math after instruction tuning, reinforcement learning, or knowledge distillation. We ask whether these gains come from major changes in the transformer layers or from smaller adjustments that keep the original structure. Using layer-wise ablation on base and trained variants, we find that math reasoning depends on a few critical layers, which stay important across all post-training methods. Removing these layers reduces math accuracy by as much as 80%, whereas factual recall tasks only show relatively smaller drops. This suggests that specialized layers for mathematical tasks form during pre-training and remain stable afterward. As measured by Normalized Mutual Information (NMI), we find that near these critical layers, tokens drift from their original syntactic clusters toward representations aligned with tokens less syntactically related but potentially more useful for downstream task.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### A Lightweight Transformer with Phase-Only Cross-Attention for Illumination-Invariant Biometric Authentication
**Date:** 2025-08-15 | **Arxiv:** [2412.19160](https://arxiv.org/abs/2412.19160)

#### Abstract
Traditional biometric systems have encountered significant setbacks due to various unavoidable factors, for example, wearing of face masks in face recognition-based biometrics and hygiene concerns in fingerprint-based biometrics. This paper proposes a novel lightweight vision transformer with phase-only cross-attention (POC-ViT) using dual biometric traits of forehead and periocular portions of the face, capable of performing well even with face masks and without any physical touch, offering a promising alternative to traditional methods. The POC-ViT framework is designed to handle two biometric traits and to capture inter-dependencies in terms of relative structural patterns. Each channel consists of a Cross-Attention using phase-only correlation (POC) that captures both their individual and correlated structural patterns. The computation of cross-attention using POC extracts the phase correlation in the spatial features. Therefore, it is robust against variations in resolution and intensity, as well as illumination changes in the input images. The lightweight model is suitable for edge device deployment. The performance of the proposed framework was successfully demonstrated using the Forehead Subcutaneous Vein Pattern and Periocular Biometric Pattern (FSVP-PBP) database, having 350 subjects. The POC-ViT framework outperformed state-of-the-art methods with an outstanding classification accuracy of $98.8\%$ with the dual biometric traits.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception
**Date:** 2025-11-26 | **Arxiv:** [2511.19509](https://arxiv.org/abs/2511.19509)

#### Abstract
Traditional vision-based material perception methods often experience substantial performance degradation under visually impaired conditions, thereby motivating the shift toward non-visual multimodal material perception. Despite this, existing approaches frequently perform naive fusion of multimodal inputs, overlooking key challenges such as modality-specific noise, missing modalities common in real-world scenarios, and the dynamically varying importance of each modality depending on the task. These limitations lead to suboptimal performance across several benchmark tasks. In this paper, we propose a robust multimodal fusion framework, TouchFormer. Specifically, we employ a Modality-Adaptive Gating (MAG) mechanism and intra- and inter-modality attention mechanisms to adaptively integrate cross-modal features, enhancing model robustness. Additionally, we introduce a Cross-Instance Embedding Regularization(CER) strategy, which significantly improves classification accuracy in fine-grained subcategory material recognition tasks. Experimental results demonstrate that, compared to existing non-visual methods, the proposed TouchFormer framework achieves classification accuracy improvements of 2.48% and 6.83% on SSMC and USMC tasks, respectively. Furthermore, real-world robotic experiments validate TouchFormer's effectiveness in enabling robots to better perceive and interpret their environment, paving the way for its deployment in safety-critical applications such as emergency response and industrial automation. The code and datasets will be open-source, and the videos are available in the supplementary materials.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Self-Attention as Distributional Projection: A Unified Interpretation of Transformer Architecture
**Date:** 2025-11-19 | **Arxiv:** [2511.13780](https://arxiv.org/abs/2511.13780)

#### Abstract
This paper presents a mathematical interpretation of self-attention by connecting it to distributional semantics principles. We show that self-attention emerges from projecting corpus-level co-occurrence statistics into sequence context. Starting from the co-occurrence matrix underlying GloVe embeddings, we demonstrate how the projection naturally captures contextual influence, with the query-key-value mechanism arising as the natural asymmetric extension for modeling directional relationships. Positional encodings and multi-head attention then follow as structured refinements of this same projection principle. Our analysis demonstrates that the Transformer architecture's particular algebraic form follows from these projection principles rather than being an arbitrary design choice.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training
**Date:** 2025-11-11 | **Arxiv:** [2511.07372](https://arxiv.org/abs/2511.07372)

#### Abstract
Recent curriculum techniques in the post-training stage of LLMs have been widely observed to outperform non-curriculum approaches in enhancing reasoning performance, yet a principled understanding of why and to what extent they work remains elusive. To address this gap, we develop a theoretical framework grounded in the intuition that progressively learning through manageable steps is more efficient than directly tackling a hard reasoning task, provided each stage stays within the model's effective competence. Under mild complexity conditions linking consecutive curriculum stages, we show that curriculum post-training avoids the exponential complexity bottleneck.   To substantiate this result, drawing insights from the Chain-of-Thoughts (CoTs) solving mathematical problems such as Countdown and parity, we model CoT generation as a states-conditioned autoregressive reasoning tree, define a uniform-branching base model to capture pretrained behavior, and formalize curriculum stages as either depth-increasing (longer reasoning chains) or hint-decreasing (shorter prefixes) subtasks. Our analysis shows that, under outcome-only reward signals, reinforcement learning finetuning achieves high accuracy with polynomial sample complexity, whereas direct learning suffers from an exponential bottleneck. We further establish analogous guarantees for test-time scaling, where curriculum-aware querying reduces both reward oracle calls and sampling cost from exponential to polynomial order.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding
**Date:** 2025-10-22 | **Arxiv:** [2510.18360](https://arxiv.org/abs/2510.18360)

#### Abstract
The performance of a deep learning model on a specific task and dataset depends heavily on its neural architecture, motivating considerable efforts to rapidly and accurately identify architectures suited to the target task and dataset. To achieve this, researchers use machine learning models-typically neural architecture encoders-to predict the performance of a neural architecture. Many state-of-the-art encoders aim to capture information flow within a neural architecture, which reflects how information moves through the forward pass and backpropagation, via a specialized model structure. However, due to their complicated structures, these flow-based encoders are significantly slower to process neural architectures compared to simpler encoders, presenting a notable practical challenge. To address this, we propose FGP, a novel pre-training method for neural architecture encoding that trains an encoder to capture the information flow without requiring specialized model structures. FGP trains an encoder to reconstruct a flow surrogate, our proposed representation of the neural architecture's information flow. Our experiments show that FGP boosts encoder performance by up to 106% in Precision-1%, compared to the same encoder trained solely with supervised learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, due to their complicated structures, these flow-based encoders are significantly slower to process neural architectures compared to simpler encoders, presenting a notable practical challenge.
* **Signal Tags:** #ai

---


### The Markovian Thinker: Architecture-Agnostic Linear Scaling of Reasoning
**Date:** 2025-10-09 | **Arxiv:** [2510.06557](https://arxiv.org/abs/2510.06557)

#### Abstract
Reinforcement learning (RL) has recently become a strong recipe for training reasoning LLMs that produce long chains of thought (LongCoT). Yet the standard RL "thinking environment", where the state is the prompt plus all prior reasoning tokens, makes the state unbounded and forces attention-based policies to pay quadratic compute as thoughts lengthen. We revisit the environment itself. We propose Markovian Thinking, a paradigm in which the policy advances reasoning while conditioning on a constant-size state, decoupling thinking length from context size. As an immediate consequence this yields linear compute with constant memory. We instantiate this idea with Delethink, an RL environment that structures reasoning into fixed-size chunks. Within each chunk, the model thinks as usual; at the boundary, the environment resets the context and reinitializes the prompt with a short carryover. Through RL, the policy learns to write a textual state near the end of each chunk sufficient for seamless continuation of reasoning after reset. Trained in this environment, an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL trained with a 24K budget. With test-time scaling, Delethink continues to improve where LongCoT plateaus. The effect of linear compute is substantial: we empirically estimate at 96K average thinking length LongCoT-RL costs 27 H100-months vs. 7 for Delethink. Analysis at RL initialization shows off-the-shelf reasoning models (1.5B-120B) often sample Markovian traces zero-shot across diverse benchmarks, providing positive samples that make RL effective at scale. Our results show that redesigning the thinking environment is a powerful lever: it enables very long reasoning without quadratic overhead and opens a path toward efficient, scalable reasoning LLMs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Disentangling Recall and Reasoning in Transformer Models through Layer-wise Attention and Activation Analysis
**Date:** 2025-10-07 | **Arxiv:** [2510.03366](https://arxiv.org/abs/2510.03366)

#### Abstract
Transformer-based language models excel at both recall (retrieving memorized facts) and reasoning (performing multi-step inference), but whether these abilities rely on distinct internal mechanisms remains unclear. Distinguishing recall from reasoning is crucial for predicting model generalization, designing targeted evaluations, and building safer interventions that affect one ability without disrupting the other.We approach this question through mechanistic interpretability, using controlled datasets of synthetic linguistic puzzles to probe transformer models at the layer, head, and neuron level. Our pipeline combines activation patching and structured ablations to causally measure component contributions to each task type. Across two model families (Qwen and LLaMA), we find that interventions on distinct layers and attention heads lead to selective impairments: disabling identified "recall circuits" reduces fact-retrieval accuracy by up to 15\% while leaving reasoning intact, whereas disabling "reasoning circuits" reduces multi-step inference by a comparable margin. At the neuron level, we observe task-specific firing patterns, though these effects are less robust, consistent with neuronal polysemanticity.Our results provide the first causal evidence that recall and reasoning rely on separable but interacting circuits in transformer models. These findings advance mechanistic interpretability by linking circuit-level structure to functional specialization and demonstrate how controlled datasets and causal interventions can yield mechanistic insights into model cognition, informing safer deployment of large language models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Estimating Deep Learning energy consumption based on model architecture and training environment
**Date:** 2025-09-26 | **Arxiv:** [2307.05520](https://arxiv.org/abs/2307.05520)

#### Abstract
To raise awareness of the environmental impact of deep learning (DL), many studies estimate the energy use of DL systems. However, energy estimates during DL training often rely on unverified assumptions. This work addresses that gap by investigating how model architecture and training environment affect energy consumption. We train a variety of computer vision models and collect energy consumption and accuracy metrics to analyze their trade-offs across configurations. Our results show that selecting the right model-training environment combination can reduce training energy consumption by up to 80.68% with less than 2% loss in $F_1$ score. We find a significant interaction effect between model and training environment: energy efficiency improves when GPU computational power scales with model complexity. Moreover, we demonstrate that common estimation practices, such as using FLOPs or GPU TDP, fail to capture these dynamics and can lead to substantial errors. To address these shortcomings, we propose the Stable Training Epoch Projection (STEP) and the Pre-training Regression-based Estimation (PRE) methods. Across evaluations, our methods outperform existing tools by a factor of two or more in estimation accuracy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, energy estimates during DL training often rely on unverified assumptions.
* **Signal Tags:** #ai

---


### LAPA: Log-Domain Prediction-Driven Dynamic Sparsity Accelerator for Transformer Model
**Date:** 2025-12-10 | **Arxiv:** [2512.07855](https://arxiv.org/abs/2512.07855)

#### Abstract
Attention-based Transformers have revolutionized natural language processing (NLP) and shown strong performance in computer vision (CV) tasks. However, as the input sequence varies, the computational bottlenecks in Transformer models exhibit dynamic behavior across stages, which calls for a cross-stage sparse acceleration strategy. Unfortunately, most existing sparse Transformer approaches are single-stage based, and their sparsity prediction mechanisms lead to significant power overhead when applied across multiple stages. To this end, this paper proposes a log-domain attention prediction algorithm-architecture co-design, named LAPA. First, an asymmetric leading one computing (ALOC) scheme is designed to eliminate expensive multiplications. Next, a mixed-precision multi-round shifting accumulation (MRSA) mechanism is further proposed to mitigate the accumulation overhead. A data-feature dependent filter (DDF) strategy is designed to work in concert with the MRSA process. Finally, an elaborate accelerator is designed to translate the theoretical enhancement into practical hardware improvement. Experimental results show that LAPA achieves 3.52x, 3.24x and 2.79x higher energy efficiency than the state-of-the-art (SOTA) works Spatten, Sanger and FACT, respectively.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, as the input sequence varies, the computational bottlenecks in Transformer models exhibit dynamic behavior across stages, which calls for a cross-stage sparse acceleration strategy.
* **Signal Tags:** #ai

---


### MGAS: Multi-Granularity Architecture Search for Trade-Off Between Model Effectiveness and Efficiency
**Date:** 2025-11-26 | **Arxiv:** [2310.15074](https://arxiv.org/abs/2310.15074)

#### Abstract
Neural architecture search (NAS) has gained significant traction in automating the design of neural networks. To reduce search time, differentiable architecture search (DAS) reframes the traditional paradigm of discrete candidate sampling and evaluation into a differentiable optimization over a super-net, followed by discretization. However, most existing DAS methods primarily focus on optimizing the coarse-grained operation-level topology, while neglecting finer-grained structures such as filter-level and weight-level patterns. This limits their ability to balance model performance with model size. Additionally, many methods compromise search quality to save memory during the search process. To tackle these issues, we propose Multi-Granularity Differentiable Architecture Search (MG-DARTS), a unified framework which aims to discover both effective and efficient architectures from scratch by comprehensively yet memory-efficiently exploring a multi-granularity search space. Specifically, we improve the existing DAS methods in two aspects. First, we adaptively adjust the retention ratios of searchable units across different granularity levels through adaptive pruning, which is achieved by learning granularity-specific discretization functions along with the evolving architecture. Second, we decompose the super-net optimization and discretization into multiple stages, each operating on a sub-net, and introduce progressive re-evaluation to enable re-pruning and regrowth of previous units, thereby mitigating potential bias. Extensive experiments on CIFAR-10, CIFAR-100 and ImageNet demonstrate that MG-DARTS outperforms other state-of-the-art methods in achieving a better trade-off between model accuracy and parameter efficiency. Codes are available at https://github.com/lxy12357/MG_DARTS.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, most existing DAS methods primarily focus on optimizing the coarse-grained operation-level topology, while neglecting finer-grained structures such as filter-level and weight-level patterns.
* **Signal Tags:** #ai

---


### Learning to Solve Weighted Maximum Satisfiability with a Co-Training Architecture
**Date:** 2025-11-26 | **Arxiv:** [2511.19544](https://arxiv.org/abs/2511.19544)

#### Abstract
Wepropose SplitGNN, a graph neural network (GNN)-based   approach that learns to solve weighted maximum satisfiabil ity (MaxSAT) problem. SplitGNN incorporates a co-training   architecture consisting of supervised message passing mech anism and unsupervised solution boosting layer. A new graph   representation called edge-splitting factor graph is proposed   to provide more structural information for learning, which is   based on spanning tree generation and edge classification. To   improve the solutions on challenging and weighted instances,   we implement a GPU-accelerated layer applying efficient   score calculation and relaxation-based optimization. Exper iments show that SplitGNN achieves 3* faster convergence   and better predictions compared with other GNN-based ar chitectures. More notably, SplitGNN successfully finds solu tions that outperform modern heuristic MaxSAT solvers on   much larger and harder weighted MaxSAT benchmarks, and   demonstrates exceptional generalization abilities on diverse   structural instances.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Neural Architecture Search for Quantum Autoencoders
**Date:** 2025-11-25 | **Arxiv:** [2511.19246](https://arxiv.org/abs/2511.19246)

#### Abstract
In recent years, machine learning and deep learning have driven advances in domains such as image classification, speech recognition, and anomaly detection by leveraging multi-layer neural networks to model complex data. Simultaneously, quantum computing (QC) promises to address classically intractable problems via quantum parallelism, motivating research in quantum machine learning (QML). Among QML techniques, quantum autoencoders show promise for compressing high-dimensional quantum and classical data. However, designing effective quantum circuit architectures for quantum autoencoders remains challenging due to the complexity of selecting gates, arranging circuit layers, and tuning parameters.   This paper proposes a neural architecture search (NAS) framework that automates the design of quantum autoencoders using a genetic algorithm (GA). By systematically evolving variational quantum circuit (VQC) configurations, our method seeks to identify high-performing hybrid quantum-classical autoencoders for data reconstruction without becoming trapped in local minima. We demonstrate effectiveness on image datasets, highlighting the potential of quantum autoencoders for efficient feature extraction within a noise-prone, near-term quantum era. Our approach lays a foundation for broader application of genetic algorithms to quantum architecture search, aiming for a robust, automated method that can adapt to varied data and hardware constraints.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, designing effective quantum circuit architectures for quantum autoencoders remains challenging due to the complexity of selecting gates, arranging circuit layers, and tuning parameters.
* **Signal Tags:** #ai

---


### Learning in Compact Spaces with Approximately Normalized Transformer
**Date:** 2025-11-20 | **Arxiv:** [2505.22014](https://arxiv.org/abs/2505.22014)

#### Abstract
The successful training of deep neural networks requires addressing challenges such as overfitting, numerical instabilities leading to divergence, and increasing variance in the residual stream. A common solution is to apply regularization and normalization techniques that usually require tuning additional hyperparameters. An alternative is to force all parameters and representations to lie on a hypersphere. This removes the need for regularization and increases convergence speed, but comes with additional costs. In this work, we propose a more holistic, approximate normalization via simple scalar multiplications motivated by the tight concentration of the norms of high-dimensional random vectors. Additionally, instead of applying strict normalization for the parameters, we constrain their norms. These modifications remove the need for weight decay and learning rate warm-up as well, but do not increase the total number of normalization layers. Our experiments with transformer architectures show up to 40% faster convergence compared to GPT models with QK normalization, with only 3% additional runtime cost. When deriving scaling laws, we found that our method enables training with larger batch sizes while preserving the favorable scaling characteristics of classic GPT architectures.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### DPL: Decoupled Prototype Learning for Enhancing Robustness of Vision-Language Transformers to Missing Modalities
**Date:** 2025-11-18 | **Arxiv:** [2505.08283](https://arxiv.org/abs/2505.08283)

#### Abstract
The performance of Visio-Language Transformers drops sharply when an input modality (e.g., image) is missing, because the model is forced to make predictions using incomplete information. Existing missing-aware prompt methods help reduce this degradation, but they still rely on conventional prediction heads (e.g., a Fully-Connected layer) that compute class scores in the same way regardless of which modality is present or absent. We introduce Decoupled Prototype Learning (DPL), a new prediction head architecture that explicitly adjusts its decision process to the observed input modalities. For each class, DPL selects a set of prototypes specific to the current missing-modality cases (image-missing, text-missing, or mixed-missing). Each prototype is then decomposed into image-specific and text-specific components, enabling the head to make decisions that depend on the information actually present. This adaptive design allows DPL to handle inputs with missing modalities more effectively while remaining fully compatible with existing prompt-based frameworks. Extensive experiments on MM-IMDb, UPMC Food-101, and Hateful Memes demonstrate that DPL outperforms state-of-the-art approaches across all widely used multimodal imag-text datasets and various missing cases.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Soft Conflict-Resolution Decision Transformer for Offline Multi-Task Reinforcement Learning
**Date:** 2025-11-18 | **Arxiv:** [2511.13133](https://arxiv.org/abs/2511.13133)

#### Abstract
Multi-task reinforcement learning (MTRL) seeks to learn a unified policy for diverse tasks, but often suffers from gradient conflicts across tasks. Existing masking-based methods attempt to mitigate such conflicts by assigning task-specific parameter masks. However, our empirical study shows that coarse-grained binary masks have the problem of over-suppressing key conflicting parameters, hindering knowledge sharing across tasks. Moreover, different tasks exhibit varying conflict levels, yet existing methods use a one-size-fits-all fixed sparsity strategy to keep training stability and performance, which proves inadequate. These limitations hinder the model's generalization and learning efficiency.   To address these issues, we propose SoCo-DT, a Soft Conflict-resolution method based by parameter importance. By leveraging Fisher information, mask values are dynamically adjusted to retain important parameters while suppressing conflicting ones. In addition, we introduce a dynamic sparsity adjustment strategy based on the Interquartile Range (IQR), which constructs task-specific thresholding schemes using the distribution of conflict and harmony scores during training. To enable adaptive sparsity evolution throughout training, we further incorporate an asymmetric cosine annealing schedule to continuously update the threshold. Experimental results on the Meta-World benchmark show that SoCo-DT outperforms the state-of-the-art method by 7.6% on MT50 and by 10.5% on the suboptimal dataset, demonstrating its effectiveness in mitigating gradient conflicts and improving overall multi-task performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, our empirical study shows that coarse-grained binary masks have the problem of over-suppressing key conflicting parameters, hindering knowledge sharing across tasks.
* **Signal Tags:** #ai

---


### Physics informed Transformer-VAE for biophysical parameter estimation: PROSAIL model inversion in Sentinel-2 imagery
**Date:** 2025-11-14 | **Arxiv:** [2511.10387](https://arxiv.org/abs/2511.10387)

#### Abstract
Accurate retrieval of vegetation biophysical variables from satellite imagery is crucial for ecosystem monitoring and agricultural management. In this work, we propose a physics-informed Transformer-VAE architecture to invert the PROSAIL radiative transfer model for simultaneous estimation of key canopy parameters from Sentinel-2 data. Unlike previous hybrid approaches that require real satellite images for self-supevised training. Our model is trained exclusively on simulated data, yet achieves performance on par with state-of-the-art methods that utilize real imagery. The Transformer-VAE incorporates the PROSAIL model as a differentiable physical decoder, ensuring that inferred latent variables correspond to physically plausible leaf and canopy properties. We demonstrate retrieval of leaf area index (LAI) and canopy chlorophyll content (CCC) on real-world field datasets (FRM4Veg and BelSAR) with accuracy comparable to models trained with real Sentinel-2 data. Our method requires no in-situ labels or calibration on real images, offering a cost-effective and self-supervised solution for global vegetation monitoring. The proposed approach illustrates how integrating physical models with advanced deep networks can improve the inversion of RTMs, opening new prospects for large-scale, physically-constrained remote sensing of vegetation traits.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Heuristic Transformer: Belief Augmented In-Context Reinforcement Learning
**Date:** 2025-11-14 | **Arxiv:** [2511.10251](https://arxiv.org/abs/2511.10251)

#### Abstract
Transformers have demonstrated exceptional in-context learning (ICL) capabilities, enabling applications across natural language processing, computer vision, and sequential decision-making. In reinforcement learning, ICL reframes learning as a supervised problem, facilitating task adaptation without parameter updates. Building on prior work leveraging transformers for sequential decision-making, we propose Heuristic Transformer (HT), an in-context reinforcement learning (ICRL) approach that augments the in-context dataset with a belief distribution over rewards to achieve better decision-making. Using a variational auto-encoder (VAE), a low-dimensional stochastic variable is learned to represent the posterior distribution over rewards, which is incorporated alongside an in-context dataset and query states as prompt to the transformer policy. We assess the performance of HT across the Darkroom, Miniworld, and MuJoCo environments, showing that it consistently surpasses comparable baselines in terms of both effectiveness and generalization. Our method presents a promising direction to bridge the gap between belief-based augmentations and transformer-based decision-making.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### ForAug: Recombining Foregrounds and Backgrounds to Improve Vision Transformer Training with Bias Mitigation
**Date:** 2025-11-14 | **Arxiv:** [2503.09399](https://arxiv.org/abs/2503.09399)

#### Abstract
Transformers, particularly Vision Transformers (ViTs), have achieved state-of-the-art performance in large-scale image classification. However, they often require large amounts of data and can exhibit biases, such as center or size bias, that limit their robustness and generalizability. This paper introduces ForAug, a novel data augmentation operation that addresses these challenges by explicitly imposing invariances into the training data, which are otherwise part of the neural network architecture. ForAug is constructed by using pretrained foundation models to separate and recombine foreground objects with different backgrounds. This recombination step enables us to take fine-grained control over object position and size, as well as background selection. We demonstrate that using ForAug significantly improves the accuracy of ViTs and other architectures by up to 4.5 percentage points (p.p.) on ImageNet, which translates to 7.3 p.p. on downstream tasks. Importantly, ForAug not only improves accuracy but also opens new ways to analyze model behavior and quantify biases. Namely, we introduce metrics for background robustness, foreground focus, center bias, and size bias and show that using ForAug during training substantially reduces these biases. In summary, ForAug provides a valuable tool for analyzing and mitigating biases, enabling the development of more robust and reliable computer vision models. Our code and dataset are publicly available at https://github.com/tobna/ForAug.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, they often require large amounts of data and can exhibit biases, such as center or size bias, that limit their robustness and generalizability.
* **Signal Tags:** #ai

---


### Adaptive Graph Learning with Transformer for Multi-Reservoir Inflow Prediction
**Date:** 2025-11-12 | **Arxiv:** [2511.07649](https://arxiv.org/abs/2511.07649)

#### Abstract
Reservoir inflow prediction is crucial for water resource management, yet existing approaches mainly focus on single-reservoir models that ignore spatial dependencies among interconnected reservoirs. We introduce AdaTrip as an adaptive, time-varying graph learning framework for multi-reservoir inflow forecasting. AdaTrip constructs dynamic graphs where reservoirs are nodes with directed edges reflecting hydrological connections, employing attention mechanisms to automatically identify crucial spatial and temporal dependencies. Evaluation on thirty reservoirs in the Upper Colorado River Basin demonstrates superiority over existing baselines, with improved performance for reservoirs with limited records through parameter sharing. Additionally, AdaTrip provides interpretable attention maps at edge and time-step levels, offering insights into hydrological controls to support operational decision-making. Our code is available at https://github.com/humphreyhuu/AdaTrip.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Galactification: painting galaxies onto dark matter only simulations using a transformer-based model
**Date:** 2025-11-12 | **Arxiv:** [2511.08438](https://arxiv.org/abs/2511.08438)

#### Abstract
Connecting the formation and evolution of galaxies to the large-scale structure is crucial for interpreting cosmological observations. While hydrodynamical simulations accurately model the correlated properties of galaxies, they are computationally prohibitive to run over volumes that match modern surveys. We address this by developing a framework to rapidly generate mock galaxy catalogs conditioned on inexpensive dark-matter-only simulations. We present a multi-modal, transformer-based model that takes 3D dark matter density and velocity fields as input, and outputs a corresponding point cloud of galaxies with their physical properties. We demonstrate that our trained model faithfully reproduces a variety of galaxy summary statistics and correctly captures their variation with changes in the underlying cosmological and astrophysical parameters, making it the first accelerated forward model to capture all the relevant galaxy properties, their full spatial distribution, and their conditional dependencies in hydrosimulations.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Attention on flow control: transformer-based reinforcement learning for lift regulation in highly disturbed flows
**Date:** 2025-11-11 | **Arxiv:** [2506.10153](https://arxiv.org/abs/2506.10153)

#### Abstract
A linear flow control strategy designed for weak disturbances may not remain effective in sequences of strong disturbances due to nonlinear interactions, but it is sensible to leverage it for developing a better strategy. In the present study, we propose a transformer-based reinforcement learning (RL) framework to learn an effective control strategy for regulating aerodynamic lift in arbitrarily long gust sequences via pitch control. The random gusts produce intermittent, high-variance flows observed only through limited surface pressure sensors, making this control problem inherently challenging compared to stationary flows. The transformer addresses the challenge of partial observability from the limited surface pressures. We demonstrate that the training can be accelerated with two techniques -- pretraining with an expert policy (here, linear control) and task-level transfer learning (here, extending a policy trained on isolated gusts to multiple gusts). We show that the learned strategy outperforms the best proportional control, with the performance gap widening as the number of gusts increases. The control strategy learned in an environment with a small number of successive gusts is shown to effectively generalize to an environment with an arbitrarily long sequence of gusts. We investigate the pivot configuration and show that quarter-chord pitching control can achieve superior lift regulation with substantially less control effort compared to mid-chord pitching control. Through a decomposition of the lift, we attribute this advantage to the dominant added-mass contribution accessible via quarter-chord pitching.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### A Hybrid Autoencoder-Transformer Model for Robust Day-Ahead Electricity Price Forecasting under Extreme Conditions
**Date:** 2025-11-11 | **Arxiv:** [2511.06898](https://arxiv.org/abs/2511.06898)

#### Abstract
Accurate day-ahead electricity price forecasting (DAEPF) is critical for the efficient operation of power systems, but extreme condition and market anomalies pose significant challenges to existing forecasting methods. To overcome these challenges, this paper proposes a novel hybrid deep learning framework that integrates a Distilled Attention Transformer (DAT) model and an Autoencoder Self-regression Model (ASM). The DAT leverages a self-attention mechanism to dynamically assign higher weights to critical segments of historical data, effectively capturing both long-term trends and short-term fluctuations. Concurrently, the ASM employs unsupervised learning to detect and isolate anomalous patterns induced by extreme conditions, such as heavy rain, heat waves, or human festivals. Experiments on datasets sampled from California and Shandong Province demonstrate that our framework significantly outperforms state-of-the-art methods in prediction accuracy, robustness, and computational efficiency. Our framework thus holds promise for enhancing grid resilience and optimizing market operations in future power systems.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Zero-shot data citation function classification using transformer-based large language models (LLMs)
**Date:** 2025-11-06 | **Arxiv:** [2511.02936](https://arxiv.org/abs/2511.02936)

#### Abstract
Efforts have increased in recent years to identify associations between specific datasets and the scientific literature that incorporates them. Knowing that a given publication cites a given dataset, the next logical step is to explore how or why that data was used. Advances in recent years with pretrained, transformer-based large language models (LLMs) offer potential means for scaling the description of data use cases in the published literature. This avoids expensive manual labeling and the development of training datasets for classical machine-learning (ML) systems. In this work we apply an open-source LLM, Llama 3.1-405B, to generate structured data use case labels for publications known to incorporate specific genomic datasets. We also introduce a novel evaluation framework for determining the efficacy of our methods. Our results demonstrate that the stock model can achieve an F1 score of .674 on a zero-shot data citation classification task with no previously defined categories. While promising, our results are qualified by barriers related to data availability, prompt overfitting, computational infrastructure, and the expense required to conduct responsible performance evaluation.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Explainable Graph Neural Architecture Search via Monte-Carlo Tree Search (Full version)
**Date:** 2025-11-05 | **Arxiv:** [2308.15734](https://arxiv.org/abs/2308.15734)

#### Abstract
The number of graph neural network (GNN) architectures has increased rapidly due to the growing adoption of graph analysis. Although we use GNNs in wide application scenarios, it is a laborious task to design/select optimal GNN architectures in diverse graphs. To reduce human efforts, graph neural architecture search (Graph NAS) has been used to search for a sub-optimal GNN architecture that combines existing components. However, existing Graph NAS methods lack explainability to understand the reasons why the model architecture is selected because they use complex search space and neural models to select architecture. Therefore, we propose an explainable Graph NAS method, called ExGNAS, which consists of (i) a simple search space that can adapt to various graphs and (ii) a search algorithm with Monte-Carlo tree that makes the decision process explainable. The combination of our search space and algorithm achieves finding accurate GNN models and the important functions within the search space. We comprehensively evaluate ExGNAS compared with four state-of-the-art Graph NAS methods in twelve graphs. Our experimental results show that ExGNAS achieves high average accuracy and efficiency; improving accuracy up to 26.1% and reducing run time up to 88%. Furthermore, we show the effectiveness of explainability by questionnaire-based user study and architecture analysis.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, existing Graph NAS methods lack explainability to understand the reasons why the model architecture is selected because they use complex search space and neural models to select architecture.
* **Signal Tags:** #ai

---


### Neural Architecture Search for global multi-step Forecasting of Energy Production Time Series
**Date:** 2025-11-04 | **Arxiv:** [2511.00035](https://arxiv.org/abs/2511.00035)

#### Abstract
The dynamic energy sector requires both predictive accuracy and runtime efficiency for short-term forecasting of energy generation under operational constraints, where timely and precise predictions are crucial. The manual configuration of complex methods, which can generate accurate global multi-step predictions without suffering from a computational bottleneck, represents a procedure with significant time requirements and high risk for human-made errors. A further intricacy arises from the temporal dynamics present in energy-related data. Additionally, the generalization to unseen data is imperative for continuously deploying forecasting techniques over time. To overcome these challenges, in this research, we design a neural architecture search (NAS)-based framework for the automated discovery of time series models that strike a balance between computational efficiency, predictive performance, and generalization power for the global, multi-step short-term forecasting of energy production time series. In particular, we introduce a search space consisting only of efficient components, which can capture distinctive patterns of energy time series. Furthermore, we formulate a novel objective function that accounts for performance generalization in temporal context and the maximal exploration of different regions of our high-dimensional search space. The results obtained on energy production time series show that an ensemble of lightweight architectures discovered with NAS outperforms state-of-the-art techniques, such as Transformers, as well as pre-trained forecasting models, in terms of both efficiency and accuracy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Stiff Circuit System Modeling via Transformer
**Date:** 2025-10-30 | **Arxiv:** [2510.24727](https://arxiv.org/abs/2510.24727)

#### Abstract
Accurate and efficient circuit behavior modeling is a cornerstone of modern electronic design automation. Among different types of circuits, stiff circuits are challenging to model using previous frameworks. In this work, we propose a new approach using Crossformer, which is a current state-of-the-art Transformer model for time-series prediction tasks, combined with Kolmogorov-Arnold Networks (KANs), to model stiff circuit transient behavior. By leveraging the Crossformer's temporal representation capabilities and the enhanced feature extraction of KANs, our method achieves improved fidelity in predicting circuit responses to a wide range of input conditions. Experimental evaluations on datasets generated through SPICE simulations of analog-to-digital converter (ADC) circuits demonstrate the effectiveness of our approach, with significant reductions in training time and error rates.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Inferring Group Intent as a Cooperative Game. An NLP-based Framework for Trajectory Analysis using Graph Transformer Neural Network
**Date:** 2025-10-29 | **Arxiv:** [2510.23905](https://arxiv.org/abs/2510.23905)

#### Abstract
This paper studies group target trajectory intent as the outcome of a cooperative game where the complex-spatio trajectories are modeled using an NLP-based generative model. In our framework, the group intent is specified by the characteristic function of a cooperative game, and allocations for players in the cooperative game are specified by either the core, the Shapley value, or the nucleolus. The resulting allocations induce probability distributions that govern the coordinated spatio-temporal trajectories of the targets that reflect the group's underlying intent. We address two key questions: (1) How can the intent of a group trajectory be optimally formalized as the characteristic function of a cooperative game? (2) How can such intent be inferred from noisy observations of the targets? To answer the first question, we introduce a Fisher-information-based characteristic function of the cooperative game, which yields probability distributions that generate coordinated spatio-temporal patterns. As a generative model for these patterns, we develop an NLP-based generative model built on formal grammar, enabling the creation of realistic multi-target trajectory data. To answer the second question, we train a Graph Transformer Neural Network (GTNN) to infer group trajectory intent-expressed as the characteristic function of the cooperative game-from observational data with high accuracy. The self-attention function of the GTNN depends on the track estimates. Thus, the formulation and algorithms provide a multi-layer approach that spans target tracking (Bayesian signal processing) and the GTNN (for group intent inference).

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### An Integrated Approach to Neural Architecture Search for Deep Q-Networks
**Date:** 2025-10-24 | **Arxiv:** [2510.19872](https://arxiv.org/abs/2510.19872)

#### Abstract
The performance of deep reinforcement learning agents is fundamentally constrained by their neural network architecture, a choice traditionally made through expensive hyperparameter searches and then fixed throughout training. This work investigates whether online, adaptive architecture optimization can escape this constraint and outperform static designs. We introduce NAS-DQN, an agent that integrates a learned neural architecture search controller directly into the DRL training loop, enabling dynamic network reconfiguration based on cumulative performance feedback. We evaluate NAS-DQN against three fixed-architecture baselines and a random search control on a continuous control task, conducting experiments over multiple random seeds. Our results demonstrate that NAS-DQN achieves superior final performance, sample efficiency, and policy stability while incurring negligible computational overhead. Critically, the learned search strategy substantially outperforms both undirected random architecture exploration and poorly-chosen fixed designs, indicating that intelligent, performance-guided search is the key mechanism driving success. These findings establish that architecture adaptation is not merely beneficial but necessary for optimal sample efficiency in online deep reinforcement learning, and suggest that the design of RL agents need not be a static offline choice but can instead be seamlessly integrated as a dynamic component of the learning process itself.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Discrete Neural Flow Samplers with Locally Equivariant Transformer
**Date:** 2025-10-23 | **Arxiv:** [2505.17741](https://arxiv.org/abs/2505.17741)

#### Abstract
Sampling from unnormalised discrete distributions is a fundamental problem across various domains. While Markov chain Monte Carlo offers a principled approach, it often suffers from slow mixing and poor convergence. In this paper, we propose Discrete Neural Flow Samplers (DNFS), a trainable and efficient framework for discrete sampling. DNFS learns the rate matrix of a continuous-time Markov chain such that the resulting dynamics satisfy the Kolmogorov equation. As this objective involves the intractable partition function, we then employ control variates to reduce the variance of its Monte Carlo estimation, leading to a coordinate descent learning algorithm. To further facilitate computational efficiency, we propose locally equivaraint Transformer, a novel parameterisation of the rate matrix that significantly improves training efficiency while preserving powerful network expressiveness. Empirically, we demonstrate the efficacy of DNFS in a wide range of applications, including sampling from unnormalised distributions, training discrete energy-based models, and solving combinatorial optimisation problems.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### MEG-GPT: A transformer-based foundation model for magnetoencephalography data
**Date:** 2025-10-22 | **Arxiv:** [2510.18080](https://arxiv.org/abs/2510.18080)

#### Abstract
Modelling the complex spatiotemporal patterns of large-scale brain dynamics is crucial for neuroscience, but traditional methods fail to capture the rich structure in modalities such as magnetoencephalography (MEG). Recent advances in deep learning have enabled significant progress in other domains, such as language and vision, by using foundation models at scale. Here, we introduce MEG-GPT, a transformer based foundation model that uses time-attention and next time-point prediction. To facilitate this, we also introduce a novel data-driven tokeniser for continuous MEG data, which preserves the high temporal resolution of continuous MEG signals without lossy transformations. We trained MEG-GPT on tokenised brain region time-courses extracted from a large-scale MEG dataset (N=612, eyes-closed rest, Cam-CAN data), and show that the learnt model can generate data with realistic spatio-spectral properties, including transient events and population variability. Critically, it performs well in downstream decoding tasks, improving downstream supervised prediction task, showing improved zero-shot generalisation across sessions (improving accuracy from 0.54 to 0.59) and subjects (improving accuracy from 0.41 to 0.49) compared to a baseline methods. Furthermore, we show the model can be efficiently fine-tuned on a smaller labelled dataset to boost performance in cross-subject decoding scenarios. This work establishes a powerful foundation model for electrophysiological data, paving the way for applications in computational neuroscience and neural decoding.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### One Token Embedding Is Enough to Deadlock Your Large Reasoning Model
**Date:** 2025-10-21 | **Arxiv:** [2510.15965](https://arxiv.org/abs/2510.15965)

#### Abstract
Modern large reasoning models (LRMs) exhibit impressive multi-step problem-solving via chain-of-thought (CoT) reasoning. However, this iterative thinking mechanism introduces a new vulnerability surface. We present the Deadlock Attack, a resource exhaustion method that hijacks an LRM's generative control flow by training a malicious adversarial embedding to induce perpetual reasoning loops. Specifically, the optimized embedding encourages transitional tokens (e.g., "Wait", "But") after reasoning steps, preventing the model from concluding its answer. A key challenge we identify is the continuous-to-discrete projection gap: naïve projections of adversarial embeddings to token sequences nullify the attack. To overcome this, we introduce a backdoor implantation strategy, enabling reliable activation through specific trigger tokens. Our method achieves a 100% attack success rate across four advanced LRMs (Phi-RM, Nemotron-Nano, R1-Qwen, R1-Llama) and three math reasoning benchmarks, forcing models to generate up to their maximum token limits. The attack is also stealthy (in terms of causing negligible utility loss on benign user inputs) and remains robust against existing strategies trying to mitigate the overthinking issue. Our findings expose a critical and underexplored security vulnerability in LRMs from the perspective of reasoning (in)efficiency.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, this iterative thinking mechanism introduces a new vulnerability surface.
* **Signal Tags:** #ai

---


### Early-stopping for Transformer model training
**Date:** 2025-10-21 | **Arxiv:** [2510.16074](https://arxiv.org/abs/2510.16074)

#### Abstract
This work introduces a novel theoretical framework grounded in Random Matrix Theory (RMT) for analyzing Transformer training dynamics. We focus on the underlying mechanisms that drive performance improvements and derive principled early-stopping criteria. Empirically, we observe that the spectral density of the shallow self-attention matrix V consistently evolves into a heavy-tailed distribution. Utilizing the PL (Power Law) fit to this matrix as a probe, we demarcate training into three stages: structural exploration, heavy-tailed structure stabilization, and convergence saturation. This staging provides guidance for preliminary stopping decisions. Crucially, we propose two consistent and validation-free criteria: a quantitative metric for heavy-tailed dynamics and a novel spectral signature indicative of convergence. The strong alignment between these criteria highlights the utility of RMT for monitoring and diagnosing the progression of Transformer model training.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Indoor Localization using Compact, Telemetry-Agnostic, Transfer-Learning Enabled Decoder-Only Transformer
**Date:** 2025-10-16 | **Arxiv:** [2510.11926](https://arxiv.org/abs/2510.11926)

#### Abstract
Indoor Wi-Fi positioning remains a challenging problem due to the high sensitivity of radio signals to environmental dynamics, channel propagation characteristics, and hardware heterogeneity. Conventional fingerprinting and model-based approaches typically require labor-intensive calibration and suffer rapid performance degradation when devices, channel or deployment conditions change. In this paper, we introduce Locaris, a decoder-only large language model (LLM) for indoor localization. Locaris treats each access point (AP) measurement as a token, enabling the ingestion of raw Wi-Fi telemetry without pre-processing. By fine-tuning its LLM on different Wi-Fi datasets, Locaris learns a lightweight and generalizable mapping from raw signals directly to device location. Our experimental study comparing Locaris with state-of-the-art methods consistently shows that Locaris matches or surpasses existing techniques for various types of telemetry. Our results demonstrate that compact LLMs can serve as calibration-free regression models for indoor localization, offering scalable and robust cross-environment performance in heterogeneous Wi-Fi deployments. Few-shot adaptation experiments, using only a handful of calibration points per device, further show that Locaris maintains high accuracy when applied to previously unseen devices and deployment scenarios. This yields sub-meter accuracy with just a few hundred samples, robust performance under missing APs and supports any and all available telemetry. Our findings highlight the practical viability of Locaris for indoor positioning in the real-world scenarios, particularly in large-scale deployments where extensive calibration is infeasible.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Transformer Model Detects Antidepressant Use From a Single Night of Sleep, Unlocking an Adherence Biomarker
**Date:** 2025-10-15 | **Arxiv:** [2510.10364](https://arxiv.org/abs/2510.10364)

#### Abstract
Antidepressant nonadherence is pervasive, driving relapse, hospitalization, suicide risk, and billions in avoidable costs. Clinicians need tools that detect adherence lapses promptly, yet current methods are either invasive (serum assays, neuroimaging) or proxy-based and inaccurate (pill counts, pharmacy refills). We present the first noninvasive biomarker that detects antidepressant intake from a single night of sleep. A transformer-based model analyzes sleep data from a consumer wearable or contactless wireless sensor to infer antidepressant intake, enabling remote, effortless, daily adherence assessment at home. Across six datasets comprising 62,000 nights from >20,000 participants (1,800 antidepressant users), the biomarker achieved AUROC = 0.84, generalized across drug classes, scaled with dose, and remained robust to concomitant psychotropics. Longitudinal monitoring captured real-world initiation, tapering, and lapses. This approach offers objective, scalable adherence surveillance with potential to improve depression care and outcomes.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### What Makes Looped Transformers Perform Better Than Non-Recursive Ones (Provably)
**Date:** 2025-10-14 | **Arxiv:** [2510.10089](https://arxiv.org/abs/2510.10089)

#### Abstract
While looped transformers (termed as Looped-Attn) often outperform standard transformers (termed as Single-Attn) on complex reasoning tasks, the theoretical basis for this advantage remains underexplored. In this paper, we explain this phenomenon through the lens of loss landscape geometry, inspired by empirical observations of their distinct dynamics at both sample and Hessian levels. To formalize this, we extend the River-Valley landscape model by distinguishing between U-shaped valleys (flat) and V-shaped valleys (steep). Based on empirical observations, we conjecture that the recursive architecture of Looped-Attn induces a landscape-level inductive bias towards River-V-Valley. Theoretical derivations based on this inductive bias guarantee a better loss convergence along the river due to valley hopping, and further encourage learning about complex patterns compared to the River-U-Valley induced by Single-Attn. Building on this insight, we propose SHIFT (Staged HIerarchical Framework for Progressive Training), a staged training framework that accelerates the training process of Looped-Attn while achieving comparable performances.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### A Modality-Aware Cooperative Co-Evolutionary Framework for Multimodal Graph Neural Architecture Search
**Date:** 2025-10-10 | **Arxiv:** [2510.07325](https://arxiv.org/abs/2510.07325)

#### Abstract
Co-exploitation attacks on software vulnerabilities pose severe risks to enterprises, a threat that can be mitigated by analyzing heterogeneous and multimodal vulnerability data. Multimodal graph neural networks (MGNNs) are well-suited to integrate complementary signals across modalities, thereby improving attack-prediction accuracy. However, designing an effective MGNN architecture is challenging because it requires coordinating modality-specific components at each layer, which is infeasible through manual tuning. Genetic algorithm (GA)-based graph neural architecture search (GNAS) provides a natural solution, yet existing methods are confined to single modalities and overlook modality heterogeneity. To address this limitation, we propose a modality-aware cooperative co-evolutionary algorithm for multimodal graph neural architecture search, termed MACC-MGNAS. First, we develop a modality-aware cooperative co-evolution (MACC) framework under a divide-and-conquer paradigm: a coordinator partitions a global chromosome population into modality-specific gene groups, local workers evolve them independently, and the coordinator reassembles chromosomes for joint evaluation. This framework effectively captures modality heterogeneity ignored by single-modality GNAS. Second, we introduce a modality-aware dual-track surrogate (MADTS) method to reduce evaluation cost and accelerate local gene evolution. Third, we design a similarity-based population diversity indicator (SPDI) strategy to adaptively balance exploration and exploitation, thereby accelerating convergence and avoiding local optima. On a standard vulnerabilities co-exploitation (VulCE) dataset, MACC-MGNAS achieves an F1-score of 81.67% within only 3 GPU-hours, outperforming the state-of-the-art competitor by 8.7% F1 while reducing computation cost by 27%.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, designing an effective MGNN architecture is challenging because it requires coordinating modality-specific components at each layer, which is infeasible through manual tuning.
* **Signal Tags:** #ai

---


### Cocoon: A System Architecture for Differentially Private Training with Correlated Noises
**Date:** 2025-10-09 | **Arxiv:** [2510.07304](https://arxiv.org/abs/2510.07304)

#### Abstract
Machine learning (ML) models memorize and leak training data, causing serious privacy issues to data owners. Training algorithms with differential privacy (DP), such as DP-SGD, have been gaining attention as a solution. However, DP-SGD adds a noise at each training iteration, which degrades the accuracy of the trained model. To improve accuracy, a new family of approaches adds carefully designed correlated noises, so that noises cancel out each other across iterations. We performed an extensive characterization study of these new mechanisms, for the first time to the best of our knowledge, and show they incur non-negligible overheads when the model is large or uses large embedding tables. Motivated by the analysis, we propose Cocoon, a hardware-software co-designed framework for efficient training with correlated noises. Cocoon accelerates models with embedding tables through pre-computing and storing correlated noises in a coalesced format (Cocoon-Emb), and supports large models through a custom near-memory processing device (Cocoon-NMP). On a real system with an FPGA-based NMP device prototype, Cocoon improves the performance by 2.33-10.82x(Cocoon-Emb) and 1.55-3.06x (Cocoon-NMP).

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, DP-SGD adds a noise at each training iteration, which degrades the accuracy of the trained model.
* **Signal Tags:** #ai

---


### MONAQ: Multi-Objective Neural Architecture Querying for Time-Series Analysis on Resource-Constrained Devices
**Date:** 2025-10-09 | **Arxiv:** [2505.10607](https://arxiv.org/abs/2505.10607)

#### Abstract
The growing use of smartphones and IoT devices necessitates efficient time-series analysis on resource-constrained hardware, which is critical for sensing applications such as human activity recognition and air quality prediction. Recent efforts in hardware-aware neural architecture search (NAS) automate architecture discovery for specific platforms; however, none focus on general time-series analysis with edge deployment. Leveraging the problem-solving and reasoning capabilities of large language models (LLM), we propose MONAQ, a novel framework that reformulates NAS into Multi-Objective Neural Architecture Querying tasks. MONAQ is equipped with multimodal query generation for processing multimodal time-series inputs and hardware constraints, alongside an LLM agent-based multi-objective search to achieve deployment-ready models via code generation. By integrating numerical data, time-series images, and textual descriptions, MONAQ improves an LLM's understanding of time-series data. Experiments on fifteen datasets demonstrate that MONAQ-discovered models outperform both handcrafted models and NAS baselines while being more efficient.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** however, none focus on general time-series analysis with edge deployment.
* **Signal Tags:** #ai

---


### Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning
**Date:** 2025-10-07 | **Arxiv:** [2508.04581](https://arxiv.org/abs/2508.04581)

#### Abstract
Large language models (LLMs) have revolutionized AI applications, yet their high computational and memory demands hinder their widespread deployment. Existing compression techniques focus on intra-block optimizations (e.g. low-rank approximation, attention head pruning), while the repetitive layered structure of transformers implies significant inter-block redundancy - a dimension largely unexplored beyond key-value (KV) caching. Inspired by dictionary learning in CNNs, we propose a framework for structured weight sharing across transformer layers. Our approach decomposes attention projection matrices into shared dictionary atoms, reducing the attention module's parameters by 66.7% while achieving on-par performance. Unlike complex methods requiring distillation or architectural changes, MASA (Matrix Atom Sharing in Attention) operates as a drop-in replacement - trained with standard optimizers - and represents each layer's weights as linear combinations of shared matrix atoms. Experiments across scales (100M-700M parameters) show that MASA achieves better benchmark accuracy and perplexity than grouped-query attention (GQA), low-rank baselines and recently proposed Repeat-all-over/Sequential sharing at comparable parameter budgets. Ablation studies confirm robustness to the dictionary size and the efficacy of shared representations in capturing cross-layer statistical regularities. Extending to Vision Transformers (ViT), MASA matches performance metrics on image classification and detection tasks with 66.7% fewer attention parameters. By combining dictionary learning strategies with transformer efficiency, MASA offers a scalable blueprint for parameter-efficient models without sacrificing performance. Finally, we investigate the possibility of employing MASA on pretrained LLMs to reduce their number of parameters without experiencing any significant drop in their performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Toward a Robust R2D2 Paradigm for Radio-interferometric Imaging: Revisiting Deep Neural Network Training and Architecture
**Date:** 2025-10-02 | **Arxiv:** [2503.02554](https://arxiv.org/abs/2503.02554)

#### Abstract
The R2D2 Deep Neural Network (DNN) series was recently introduced for image formation in radio interferometry. It can be understood as a learned version of CLEAN, whose minor cycles are substituted with DNNs. We revisit R2D2 on the grounds of series convergence, training methodology, and DNN architecture, improving its robustness in terms of generalizability beyond training conditions, capability to deliver high data fidelity, and epistemic uncertainty. First, while still focusing on telescope-specific training, we enhance the learning process by randomizing Fourier sampling integration times, incorporating multiscan multinoise configurations, and varying imaging settings, including pixel resolution and visibility-weighting scheme. Second, we introduce a convergence criterion whereby the reconstruction process stops when the data residual is compatible with noise, rather than simply using all available DNNs. This not only increases the reconstruction efficiency by reducing its computational cost, but also refines training by pruning out the data/image pairs for which optimal data fidelity is reached before training the next DNN. Third, we substitute R2D2's early U-Net DNN with a novel architecture (U-WDSR) combining U-Net and WDSR, which leverages wide activation, dense skip connections, weight normalization, and low-rank convolution to improve feature reuse and reconstruction precision. As previously, R2D2 was trained for monochromatic intensity imaging with the Very Large Array at fixed $512 \times 512$ image size. Simulations on a wide range of inverse problems and a case study on real data reveal that the new R2D2 model consistently outperforms its earlier version in image reconstruction quality, data fidelity, and epistemic uncertainty.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### WDformer: A Wavelet-based Differential Transformer Model for Time Series Forecasting
**Date:** 2025-10-01 | **Arxiv:** [2509.25231](https://arxiv.org/abs/2509.25231)

#### Abstract
Time series forecasting has various applications, such as meteorological rainfall prediction, traffic flow analysis, financial forecasting, and operational load monitoring for various systems. Due to the sparsity of time series data, relying solely on time-domain or frequency-domain modeling limits the model's ability to fully leverage multi-domain information. Moreover, when applied to time series forecasting tasks, traditional attention mechanisms tend to over-focus on irrelevant historical information, which may introduce noise into the prediction process, leading to biased results. We proposed WDformer, a wavelet-based differential Transformer model. This study employs the wavelet transform to conduct a multi-resolution analysis of time series data. By leveraging the advantages of joint representation in the time-frequency domain, it accurately extracts the key information components that reflect the essential characteristics of the data. Furthermore, we apply attention mechanisms on inverted dimensions, allowing the attention mechanism to capture relationships between multiple variables. When performing attention calculations, we introduced the differential attention mechanism, which computes the attention score by taking the difference between two separate softmax attention matrices. This approach enables the model to focus more on important information and reduce noise. WDformer has achieved state-of-the-art (SOTA) results on multiple challenging real-world datasets, demonstrating its accuracy and effectiveness. Code is available at https://github.com/xiaowangbc/WDformer.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Impossibility of Inverse Permutation Learning in Transformer Models
**Date:** 2025-09-30 | **Arxiv:** [2509.24125](https://arxiv.org/abs/2509.24125)

#### Abstract
In this technical note, we study the problem of inverse permutation learning in decoder-only transformers. Given a permutation and a string to which that permutation has been applied, the model is tasked with producing the original (``canonical'') string. We argue that this task models a natural robustness property across a variety of reasoning tasks, including long-context retrieval, multiple choice QA and in-context learning. Our primary contribution is an impossibility result: we show that an arbitrary depth, decoder-only transformer cannot learn this task. This result concerns the expressive capacity of decoder-only transformer models and is agnostic to training dynamics or sample complexity. We give a pair of alternative constructions under which inverse permutation learning is feasible. The first of these highlights the fundamental role of the causal attention mask, and reveals a gap between the expressivity of encoder-decoder transformers and the more popular decoder-only architecture. The latter result is more surprising: we show that simply padding the input with ``scratch tokens" yields a construction under which inverse permutation learning is possible. We conjecture that this may suggest an alternative mechanism by which chain-of-thought prompting or, more generally, intermediate ``thinking'' tokens can enable reasoning in large language models, even when these tokens encode no meaningful semantic information (e.g., the results of intermediate computations).

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Innovative Deep Learning Architecture for Enhanced Altered Fingerprint Recognition
**Date:** 2025-09-26 | **Arxiv:** [2509.20537](https://arxiv.org/abs/2509.20537)

#### Abstract
Altered fingerprint recognition (AFR) is challenging for biometric verification in applications such as border control, forensics, and fiscal admission. Adversaries can deliberately modify ridge patterns to evade detection, so robust recognition of altered prints is essential. We present DeepAFRNet, a deep learning recognition model that matches and recognizes distorted fingerprint samples. The approach uses a VGG16 backbone to extract high-dimensional features and cosine similarity to compare embeddings. We evaluate on the SOCOFing Real-Altered subset with three difficulty levels (Easy, Medium, Hard). With strict thresholds, DeepAFRNet achieves accuracies of 96.7 percent, 98.76 percent, and 99.54 percent for the three levels. A threshold-sensitivity study shows that relaxing the threshold from 0.92 to 0.72 sharply degrades accuracy to 7.86 percent, 27.05 percent, and 29.51 percent, underscoring the importance of threshold selection in biometric systems. By using real altered samples and reporting per-level metrics, DeepAFRNet addresses limitations of prior work based on synthetic alterations or limited verification protocols, and indicates readiness for real-world deployments where both security and recognition resilience are critical.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning from Scratch: Structurally-masked Transformer for Next Generation Lib-free Simulation
**Date:** 2025-09-16 | **Arxiv:** [2507.17396](https://arxiv.org/abs/2507.17396)

#### Abstract
This paper proposes a neural framework for power and timing prediction of multi-stage data path, distinguishing itself from traditional lib-based analytical methods dependent on driver characterization and load simplifications. To the best of our knowledge, this is the first language-based, netlist-aware neural network designed explicitly for standard cells. Our approach employs two pre-trained neural models of waveform prediction and delay estimation that directly infer transient waveforms and propagation delays from SPICE netlists, conditioned on critical physical parameters such as load capacitance, input slew, and gate size. This method accurately captures both intrinsic and coupling-induced delay effects without requiring simplification or interpolation. For multi-stage timing prediction, we implement a recursive propagation strategy where predicted waveforms from each stage feed into subsequent stages, cumulatively capturing delays across the logic chain. This approach ensures precise timing alignment and complete waveform visibility throughout complex signal pathways. The waveform prediction utilizes a hybrid CNN-Transformer architecture with netlist-aware node-level encoding, addressing traditional Transformers' fixed input dimensionality constraints. Additionally, specialized subnetworks separately handle primary delay estimation and crosstalk correction. Experimental results demonstrate SPICE-level accuracy, consistently achieving RMSE below 0.0098 across diverse industrial circuits. The proposed framework provides a scalable, structurally adaptable neural alternative to conventional power and timing engines, demonstrating high fidelity to physical circuit behaviors.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### TransGAT: Transformer-Based Graph Neural Networks for Multi-Dimensional Automated Essay Scoring
**Date:** 2025-09-03 | **Arxiv:** [2509.01640](https://arxiv.org/abs/2509.01640)

#### Abstract
Essay writing is a critical component of student assessment, yet manual scoring is labor-intensive and inconsistent. Automated Essay Scoring (AES) offers a promising alternative, but current approaches face limitations. Recent studies have incorporated Graph Neural Networks (GNNs) into AES using static word embeddings that fail to capture contextual meaning, especially for polysemous words. Additionally, many methods rely on holistic scoring, overlooking specific writing aspects such as grammar, vocabulary, and cohesion. To address these challenges, this study proposes TransGAT, a novel approach that integrates fine-tuned Transformer models with GNNs for analytic scoring. TransGAT combines the contextual understanding of Transformers with the relational modeling strength of Graph Attention Networks (GAT). It performs two-stream predictions by pairing each fine-tuned Transformer (BERT, RoBERTa, and DeBERTaV3) with a separate GAT. In each pair, the first stream generates essay-level predictions, while the second applies GAT to Transformer token embeddings, with edges constructed from syntactic dependencies. The model then fuses predictions from both streams to produce the final analytic score. Experiments on the ELLIPSE dataset show that TransGAT outperforms baseline models, achieving an average Quadratic Weighted Kappa (QWK) of 0.854 across all analytic scoring dimensions. These findings highlight the potential of TransGAT to advance AES systems.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Beyond Imaging: Vision Transformer Digital Twin Surrogates for 3D+T Biological Tissue Dynamics
**Date:** 2025-08-25 | **Arxiv:** [2508.15883](https://arxiv.org/abs/2508.15883)

#### Abstract
Understanding the dynamic organization and homeostasis of living tissues requires high-resolution, time-resolved imaging coupled with methods capable of extracting interpretable, predictive insights from complex datasets. Here, we present the Vision Transformer Digital Twin Surrogate Network (VT-DTSN), a deep learning framework for predictive modeling of 3D+T imaging data from biological tissue. By leveraging Vision Transformers pretrained with DINO (Self-Distillation with NO Labels) and employing a multi-view fusion strategy, VT-DTSN learns to reconstruct high-fidelity, time-resolved dynamics of a Drosophila midgut while preserving morphological and feature-level integrity across imaging depths. The model is trained with a composite loss prioritizing pixel-level accuracy, perceptual structure, and feature-space alignment, ensuring biologically meaningful outputs suitable for in silico experimentation and hypothesis testing. Evaluation across layers and biological replicates demonstrates VT-DTSN's robustness and consistency, achieving low error rates and high structural similarity while maintaining efficient inference through model optimization. This work establishes VT-DTSN as a feasible, high-fidelity surrogate for cross-timepoint reconstruction and for studying tissue dynamics, enabling computational exploration of cellular behaviors and homeostasis to complement time-resolved imaging studies in biological research.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Multimodal Quantum Vision Transformer for Enzyme Commission Classification from Biochemical Representations
**Date:** 2025-08-21 | **Arxiv:** [2508.14844](https://arxiv.org/abs/2508.14844)

#### Abstract
Accurately predicting enzyme functionality remains one of the major challenges in computational biology, particularly for enzymes with limited structural annotations or sequence homology. We present a novel multimodal Quantum Machine Learning (QML) framework that enhances Enzyme Commission (EC) classification by integrating four complementary biochemical modalities: protein sequence embeddings, quantum-derived electronic descriptors, molecular graph structures, and 2D molecular image representations. Quantum Vision Transformer (QVT) backbone equipped with modality-specific encoders and a unified cross-attention fusion module. By integrating graph features and spatial patterns, our method captures key stereoelectronic interactions behind enzyme function. Experimental results demonstrate that our multimodal QVT model achieves a top-1 accuracy of 85.1%, outperforming sequence-only baselines by a substantial margin and achieving better performance results compared to other QML models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Development of Pre-Trained Transformer-based Models for the Nepali Language
**Date:** 2025-08-20 | **Arxiv:** [2411.15734](https://arxiv.org/abs/2411.15734)

#### Abstract
Transformer-based pre-trained language models have dominated the field of Natural Language Processing (NLP) for quite some time now. However, the Nepali language, spoken by approximately 32 million people worldwide, remains significantly underrepresented in this domain. This underrepresentation is primarily attributed to the scarcity of monolingual data corpora and limited available resources for the Nepali language. While existing efforts have predominantly concentrated on basic encoder-based models, there is a notable gap in the exploration of decoder-based architectures. To address this gap, we have collected 27.5 GB of Nepali text data, approximately 2.4x larger than any previously available Nepali language corpus. Leveraging this data, we pre-trained three different models i.e., BERT, RoBERTa, and GPT-2, exclusively for the Nepali Language. Furthermore, we performed instruction tuning and explored its potential for monolingual Nepali data, providing a foundation for future research. Our models outperformed the existing best model by 2 points on Nep-gLUE benchmark, scoring 95.60 and also outperformed existing models on text generation tasks, demonstrating improvements in both understanding and generating Nepali text.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, the Nepali language, spoken by approximately 32 million people worldwide, remains significantly underrepresented in this domain.
* **Signal Tags:** #ai

---


### Hebbian Physics Networks: A Self-Organizing Computational Architecture Based on Local Physical Laws
**Date:** 2025-12-10 | **Arxiv:** [2507.00641](https://arxiv.org/abs/2507.00641)

#### Abstract
Physical transport processes organize through local interactions that redistribute imbalance while preserving conservation. Classical solvers enforce this organization by applying fixed discrete operators on rigid grids. We introduce the Hebbian Physics Network (HPN), a computational framework that replaces this rigid scaffolding with a plastic transport geometry. An HPN is a coupled dynamical system of physical states on nodes and constitutive weights on edges in a graph. Residuals--local violations of continuity, momentum balance, or energy conservation--act as thermodynamic forces that drive the joint evolution of both the state and the operator (i.e. the adaptive weights). The weights adapt through a three-factor Hebbian rule, which we prove constitutes a strictly local gradient descent on the residual energy. This mechanism ensures thermodynamic stability: near equilibrium, the learned operator naturally converges to a symmetric, positive-definite form, rigorously reproducing Onsagerś reciprocal relations without explicit enforcement. Far from equilibrium, the system undergoes a self-organizing search for a transport topology that restores global coercivity. Unlike optimization-based approaches that impose physics through global loss functions, HPNs embed conservation intrinsically: transport is restored locally by the evolving operator itself, without a global Poisson solve or backpropagated objective. We demonstrate the framework on scalar diffusion and incompressible lid-driven cavity flow, showing that physically consistent transport geometries and flow structures emerge from random initial conditions solely through residual-driven local adaptation. HPNs thus reframe computation not as the solution of a fixed equation, but as a thermodynamic relaxation process where the constitutive geometry and physical state co-evolve.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
