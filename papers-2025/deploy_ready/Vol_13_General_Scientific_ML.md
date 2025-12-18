# Vol 13 General Scientific ML
*Enriched by BITCOREOS | Phase 4 Batch 3*

---

### From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback
**Date:** 2025-11-12 | **Arxiv:** [2511.08035](https://hub.bitwiki.org/t/from-sequential-to-recursive-enhancing-decision-focused-learning-with-bidirectional-feedback/22985)

#### Abstract
Decision-focused learning (DFL) has emerged as a powerful end-to-end alternative to conventional predict-then-optimize (PTO) pipelines by directly optimizing predictive models through downstream decision losses. Existing DFL frameworks are limited by their strictly sequential structure, referred to as sequential DFL (S-DFL). However, S-DFL fails to capture the bidirectional feedback between prediction and optimization in complex interaction scenarios. In view of this, we first time propose recursive decision-focused learning (R-DFL), a novel framework that introduces bidirectional feedback between downstream optimization and upstream prediction. We further extend two distinct differentiation methods: explicit unrolling via automatic differentiation and implicit differentiation based on fixed-point methods, to facilitate efficient gradient propagation in R-DFL. We rigorously prove that both methods achieve comparable gradient accuracy, with the implicit method offering superior computational efficiency. Extensive experiments on both synthetic and real-world datasets, including the newsvendor problem and the bipartite matching problem, demonstrate that R-DFL not only substantially enhances the final decision quality over sequential baselines but also exhibits robust adaptability across diverse scenarios in closed-loop decision-making problems.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, S-DFL fails to capture the bidirectional feedback between prediction and optimization in complex interaction scenarios.
* **Signal Tags:** #ai

---


### PRISM: Privacy-preserving Inference System with Homomorphic Encryption and Modular Activation
**Date:** 2025-11-12 | **Arxiv:** [2511.07807](https://hub.bitwiki.org/t/prism-privacy-preserving-inference-system-with-homomorphic-encryption-and-modular-activation/23060)

#### Abstract
With the rapid advancements in machine learning, models have become increasingly capable of learning and making predictions in various industries. However, deploying these models in critical infrastructures presents a major challenge, as concerns about data privacy prevent unrestricted data sharing. Homomorphic encryption (HE) offers a solution by enabling computations on encrypted data, but it remains incompatible with machine learning models like convolutional neural networks (CNNs), due to their reliance on non-linear activation functions. To bridge this gap, this work proposes an optimized framework that replaces standard non-linear functions with homomorphically compatible approximations, ensuring secure computations while minimizing computational overhead. The proposed approach restructures the CNN architecture and introduces an efficient activation function approximation method to mitigate the performance trade-offs introduced by encryption. Experiments on CIFAR-10 achieve 94.4% accuracy with 2.42 s per single encrypted sample and 24,000 s per 10,000 encrypted samples, using a degree-4 polynomial and Softplus activation under CKKS, balancing accuracy and privacy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, deploying these models in critical infrastructures presents a major challenge, as concerns about data privacy prevent unrestricted data sharing.
* **Signal Tags:** #ai

---


### Token Is All You Need: Cognitive Planning through Belief-Intent Co-Evolution
**Date:** 2025-11-11 | **Arxiv:** [2511.05540](https://hub.bitwiki.org/t/token-is-all-you-need-cognitive-planning-through-belief-intent-co-evolution/22709)

#### Abstract
We challenge the long-standing assumption that exhaustive scene modeling is required for high-performance end-to-end autonomous driving (E2EAD). Inspired by cognitive science, we propose that effective planning arises not from reconstructing the world, but from the co-evolution of belief and intent within a minimal set of semantically rich tokens. Experiments on the nuPlan benchmark (720 scenarios, 11k+ samples) reveal three principles: (1) sparse intent tokens alone achieve 0.487 m ADE, demonstrating strong performance without future prediction; (2) conditioning trajectory decoding on predicted future tokens reduces ADE to 0.382 m, a 21.6% improvement, showing that performance emerges from cognitive planning; and (3) explicit reconstruction loss degrades performance, confirming that task-driven belief-intent co-evolution suffices under reliable perception inputs. Crucially, we observe the emergence of cognitive consistency: through prolonged training, the model spontaneously develops stable token dynamics that balance current perception (belief) and future goals (intent). This process, accompanied by "temporal fuzziness," enables robustness under uncertainty and continuous self-optimization. Our work establishes a new paradigm: intelligence lies not in pixel fidelity, but in the tokenized duality of belief and intent. By reframing planning as understanding rather than reaction, TIWM bridges the gap between world models and VLA systems, paving the way for foresightful agents that plan through imagination. Note: Numerical comparisons with methods reporting results on nuScenes are indicative only, as nuPlan presents a more challenging planning-focused evaluation.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Comparison of Fully Homomorphic Encryption and Garbled Circuit Techniques in Privacy-Preserving Machine Learning Inference
**Date:** 2025-10-10 | **Arxiv:** [2510.07457](https://hub.bitwiki.org/t/comparison-of-fully-homomorphic-encryption-and-garbled-circuit-techniques-in-privacy-preserving-machine-learning-inference/16028)

#### Abstract
Machine Learning (ML) is making its way into fields such as healthcare, finance, and Natural Language Processing (NLP), and concerns over data privacy and model confidentiality continue to grow. Privacy-preserving Machine Learning (PPML) addresses this challenge by enabling inference on private data without revealing sensitive inputs or proprietary models. Leveraging Secure Computation techniques from Cryptography, two widely studied approaches in this domain are Fully Homomorphic Encryption (FHE) and Garbled Circuits (GC). This work presents a comparative evaluation of FHE and GC for secure neural network inference. A two-layer neural network (NN) was implemented using the CKKS scheme from the Microsoft SEAL library (FHE) and the TinyGarble2.0 framework (GC) by IntelLabs. Both implementations are evaluated under the semi-honest threat model, measuring inference output error, round-trip time, peak memory usage, communication overhead, and communication rounds. Results reveal a trade-off: modular GC offers faster execution and lower memory consumption, while FHE supports non-interactive inference.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Privacy-Preserving Federated Learning via Homomorphic Adversarial Networks
**Date:** 2025-09-09 | **Arxiv:** [2412.01650](https://hub.bitwiki.org/t/privacy-preserving-federated-learning-via-homomorphic-adversarial-networks/8559)

#### Abstract
Privacy-preserving federated learning (PPFL) aims to train a global model for multiple clients while maintaining their data privacy. However, current PPFL protocols exhibit one or more of the following insufficiencies: considerable degradation in accuracy, the requirement for sharing keys, and cooperation during the key generation or decryption processes. As a mitigation, we develop the first protocol that utilizes neural networks to implement PPFL, as well as incorporating an Aggregatable Hybrid Encryption scheme tailored to the needs of PPFL. We name these networks as Homomorphic Adversarial Networks (HANs) which demonstrate that neural networks are capable of performing tasks similar to multi-key homomorphic encryption (MK-HE) while solving the problems of key distribution and collaborative decryption. Our experiments show that HANs are robust against privacy attacks. Compared with non-private federated learning, experiments conducted on multiple datasets demonstrate that HANs exhibit a negligible accuracy loss (at most 1.35%). Compared to traditional MK-HE schemes, HANs increase encryption aggregation speed by 6,075 times while incurring a 29.2 times increase in communication overhead.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, current PPFL protocols exhibit one or more of the following insufficiencies: considerable degradation in accuracy, the requirement for sharing keys, and cooperation during the key generation or decryption processes.
* **Signal Tags:** #ai

---


### A Fast Anti-Jamming Cognitive Radar Deployment Algorithm Based on Reinforcement Learning
**Date:** 2025-12-08 | **Arxiv:** [2512.05753](https://hub.bitwiki.org/t/a-fast-anti-jamming-cognitive-radar-deployment-algorithm-based-on-reinforcement-learning/27931)

#### Abstract
The fast deployment of cognitive radar to counter jamming remains a critical challenge in modern warfare, where more efficient deployment leads to quicker detection of targets. Existing methods are primarily based on evolutionary algorithms, which are time-consuming and prone to falling into local optima. We tackle these drawbacks via the efficient inference of neural networks and propose a brand new framework: Fast Anti-Jamming Radar Deployment Algorithm (FARDA). We first model the radar deployment problem as an end-to-end task and design deep reinforcement learning algorithms to solve it, where we develop integrated neural modules to perceive heatmap information and a brand new reward format. Empirical results demonstrate that our method achieves coverage comparable to evolutionary algorithms while deploying radars approximately 7,000 times faster. Further ablation experiments confirm the necessity of each component of FARDA.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Natural Language Actor-Critic: Scalable Off-Policy Learning in Language Space
**Date:** 2025-12-05 | **Arxiv:** [2512.04601](https://hub.bitwiki.org/t/natural-language-actor-critic-scalable-off-policy-learning-in-language-space/27652)

#### Abstract
Large language model (LLM) agents -- LLMs that dynamically interact with an environment over long horizons -- have become an increasingly important area of research, enabling automation in complex tasks involving tool-use, web browsing, and dialogue with people. In the absence of expert demonstrations, training LLM agents has relied on policy gradient methods that optimize LLM policies with respect to an (often sparse) reward function. However, in long-horizon tasks with sparse rewards, learning from trajectory-level rewards can be noisy, leading to training that is unstable and has high sample complexity. Furthermore, policy improvement hinges on discovering better actions through exploration, which can be difficult when actions lie in natural language space. In this paper, we propose Natural Language Actor-Critic (NLAC), a novel actor-critic algorithm that trains LLM policies using a generative LLM critic that produces natural language rather than scalar values. This approach leverages the inherent strengths of LLMs to provide a richer and more actionable training signal; particularly, in tasks with large, open-ended action spaces, natural language explanations for why an action is suboptimal can be immensely useful for LLM policies to reason how to improve their actions, without relying on random exploration. Furthermore, our approach can be trained off-policy without policy gradients, offering a more data-efficient and stable alternative to existing on-policy methods. We present results on a mixture of reasoning, web browsing, and tool-use with dialogue tasks, demonstrating that NLAC shows promise in outperforming existing training approaches and offers a more scalable and stable training paradigm for LLM agents.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, in long-horizon tasks with sparse rewards, learning from trajectory-level rewards can be noisy, leading to training that is unstable and has high sample complexity.
* **Signal Tags:** #ai

---


### Brian Intensify: An Adaptive Machine Learning Framework for Auditory EEG Stimulation and Cognitive Enhancement in FXS
**Date:** 2025-11-14 | **Arxiv:** [2511.09765](https://hub.bitwiki.org/t/brian-intensify-an-adaptive-machine-learning-framework-for-auditory-eeg-stimulation-and-cognitive-enhancement-in-fxs/23554)

#### Abstract
Neurodevelopmental disorders such as Fragile X Syndrome (FXS) and Autism Spectrum Disorder (ASD) are characterized by disrupted cortical oscillatory activity, particularly in the alpha and gamma frequency bands. These abnormalities are linked to deficits in attention, sensory processing, and cognitive function. In this work, we present an adaptive machine learning-based brain-computer interface (BCI) system designed to modulate neural oscillations through frequency-specific auditory stimulation to enhance cognitive readiness in individuals with FXS. EEG data were recorded from 38 participants using a 128-channel system under a stimulation paradigm consisting of a 30-second baseline (no stimulus) followed by 60-second auditory entrainment episodes at 7Hz, 9Hz, 11Hz, and 13Hz. A comprehensive analysis of power spectral features (Alpha, Gamma, Delta, Theta, Beta) and cross-frequency coupling metrics (Alpha-Gamma, Alpha-Beta, etc.) was conducted. The results identified Peak Alpha Power, Peak Gamma Power, and Alpha Power per second per channel as the most discriminative biomarkers. The 13Hz stimulation condition consistently elicited a significant increase in Alpha activity and suppression of Gamma activity, aligning with our optimization objective. A supervised machine learning framework was developed to predict EEG responses and dynamically adjust stimulation parameters, enabling real-time, subject-specific adaptation. This work establishes a novel EEG-driven optimization framework for cognitive neuromodulation, providing a foundational model for next-generation AI-integrated BCI systems aimed at personalized neurorehabilitation in FXS and related disorders.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### An MLCommons Scientific Benchmarks Ontology
**Date:** 2025-11-11 | **Arxiv:** [2511.05614](https://hub.bitwiki.org/t/an-mlcommons-scientific-benchmarks-ontology/22473)

#### Abstract
Scientific machine learning research spans diverse domains and data modalities, yet existing benchmark efforts remain siloed and lack standardization. This makes novel and transformative applications of machine learning to critical scientific use-cases more fragmented and less clear in pathways to impact. This paper introduces an ontology for scientific benchmarking developed through a unified, community-driven effort that extends the MLCommons ecosystem to cover physics, chemistry, materials science, biology, climate science, and more. Building on prior initiatives such as XAI-BENCH, FastML Science Benchmarks, PDEBench, and the SciMLBench framework, our effort consolidates a large set of disparate benchmarks and frameworks into a single taxonomy of scientific, application, and system-level benchmarks. New benchmarks can be added through an open submission workflow coordinated by the MLCommons Science Working Group and evaluated against a six-category rating rubric that promotes and identifies high-quality benchmarks, enabling stakeholders to select benchmarks that meet their specific needs. The architecture is extensible, supporting future scientific and AI/ML motifs, and we discuss methods for identifying emerging computing patterns for unique scientific workloads. The MLCommons Science Benchmarks Ontology provides a standardized, scalable foundation for reproducible, cross-domain benchmarking in scientific machine learning. A companion webpage for this work has also been developed as the effort evolves: https://mlcommons-science.github.io/benchmark/

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Incorporating Quality of Life in Climate Adaptation Planning via Reinforcement Learning
**Date:** 2025-11-06 | **Arxiv:** [2511.03238](https://hub.bitwiki.org/t/incorporating-quality-of-life-in-climate-adaptation-planning-via-reinforcement-learning/21825)

#### Abstract
Urban flooding is expected to increase in frequency and severity as a consequence of climate change, causing wide-ranging impacts that include a decrease in urban Quality of Life (QoL). Meanwhile, policymakers must devise adaptation strategies that can cope with the uncertain nature of climate change and the complex and dynamic nature of urban flooding. Reinforcement Learning (RL) holds significant promise in tackling such complex, dynamic, and uncertain problems. Because of this, we use RL to identify which climate adaptation pathways lead to a higher QoL in the long term. We do this using an Integrated Assessment Model (IAM) which combines a rainfall projection model, a flood model, a transport accessibility model, and a quality of life index. Our preliminary results suggest that this approach can be used to learn optimal adaptation measures and it outperforms other realistic and real-world planning strategies. Our framework is publicly available: https://github.com/MLSM-at-DTU/maat_qol_framework.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum
**Date:** 2025-10-31 | **Arxiv:** [2505.12191](https://hub.bitwiki.org/t/ditch-the-denoiser-emergence-of-noise-robustness-in-self-supervised-learning-from-data-curriculum/20724)

#### Abstract
Self-Supervised Learning (SSL) has become a powerful solution to extract rich representations from unlabeled data. Yet, SSL research is mostly focused on clean, curated and high-quality datasets. As a result, applying SSL on noisy data remains a challenge, despite being crucial to applications such as astrophysics, medical imaging, geophysics or finance. In this work, we present a fully self-supervised framework that enables noise-robust representation learning without requiring a denoiser at inference or downstream fine-tuning. Our method first trains an SSL denoiser on noisy data, then uses it to construct a denoised-to-noisy data curriculum (i.e., training first on denoised, then noisy samples) for pretraining a SSL backbone (e.g., DINOv2), combined with a teacher-guided regularization that anchors noisy embeddings to their denoised counterparts. This process encourages the model to internalize noise robustness. Notably, the denoiser can be discarded after pretraining, simplifying deployment. On ImageNet-1k with ViT-B under extreme Gaussian noise ($σ=255$, SNR = 0.72 dB), our method improves linear probing accuracy by 4.8% over DINOv2, demonstrating that denoiser-free robustness can emerge from noise-aware pretraining. The code is available at https://github.com/wenquanlu/noisy_dinov2.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Planning and Learning in Average Risk-aware MDPs
**Date:** 2025-10-27 | **Arxiv:** [2503.17629](https://hub.bitwiki.org/t/planning-and-learning-in-average-risk-aware-mdps/19652)

#### Abstract
For continuing tasks, average cost Markov decision processes have well-documented value and can be solved using efficient algorithms. However, it explicitly assumes that the agent is risk-neutral. In this work, we extend risk-neutral algorithms to accommodate the more general class of dynamic risk measures. Specifically, we propose a relative value iteration (RVI) algorithm for planning and design two model-free Q-learning algorithms, namely a generic algorithm based on the multi-level Monte Carlo (MLMC) method, and an off-policy algorithm dedicated to utility-based shortfall risk measures. Both the RVI and MLMC-based Q-learning algorithms are proven to converge to optimality. Numerical experiments validate our analysis, confirm empirically the convergence of the off-policy algorithm, and demonstrate that our approach enables the identification of policies that are finely tuned to the intricate risk-awareness of the agent that they serve.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, it explicitly assumes that the agent is risk-neutral.
* **Signal Tags:** #ai

---


### Steering Autoregressive Music Generation with Recursive Feature Machines
**Date:** 2025-10-23 | **Arxiv:** [2510.19127](https://hub.bitwiki.org/t/steering-autoregressive-music-generation-with-recursive-feature-machines/18838)

#### Abstract
Controllable music generation remains a significant challenge, with existing methods often requiring model retraining or introducing audible artifacts. We introduce MusicRFM, a framework that adapts Recursive Feature Machines (RFMs) to enable fine-grained, interpretable control over frozen, pre-trained music models by directly steering their internal activations. RFMs analyze a model's internal gradients to produce interpretable "concept directions", or specific axes in the activation space that correspond to musical attributes like notes or chords. We first train lightweight RFM probes to discover these directions within MusicGen's hidden states; then, during inference, we inject them back into the model to guide the generation process in real-time without per-step optimization. We present advanced mechanisms for this control, including dynamic, time-varying schedules and methods for the simultaneous enforcement of multiple musical properties. Our method successfully navigates the trade-off between control and generation quality: we can increase the accuracy of generating a target musical note from 0.23 to 0.82, while text prompt adherence remains within approximately 0.02 of the unsteered baseline, demonstrating effective control with minimal impact on prompt fidelity. We release code to encourage further exploration on RFMs in the music domain.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Hyperbolic Dataset Distillation
**Date:** 2025-10-20 | **Arxiv:** [2505.24623](https://hub.bitwiki.org/t/hyperbolic-dataset-distillation/17891)

#### Abstract
To address the computational and storage challenges posed by large-scale datasets in deep learning, dataset distillation has been proposed to synthesize a compact dataset that replaces the original while maintaining comparable model performance. Unlike optimization-based approaches that require costly bi-level optimization, distribution matching (DM) methods improve efficiency by aligning the distributions of synthetic and original data, thereby eliminating nested optimization. DM achieves high computational efficiency and has emerged as a promising solution. However, existing DM methods, constrained to Euclidean space, treat data as independent and identically distributed points, overlooking complex geometric and hierarchical relationships. To overcome this limitation, we propose a novel hyperbolic dataset distillation method, termed HDD. Hyperbolic space, characterized by negative curvature and exponential volume growth with distance, naturally models hierarchical and tree-like structures. HDD embeds features extracted by a shallow network into the Lorentz hyperbolic space, where the discrepancy between synthetic and original data is measured by the hyperbolic (geodesic) distance between their centroids. By optimizing this distance, the hierarchical structure is explicitly integrated into the distillation process, guiding synthetic samples to gravitate towards the root-centric regions of the original data distribution while preserving their underlying geometric characteristics. Furthermore, we find that pruning in hyperbolic space requires only 20% of the distilled core set to retain model performance, while significantly improving training stability. To the best of our knowledge, this is the first work to incorporate the hyperbolic space into the dataset distillation process. The code is available at https://github.com/Guang000/HDD.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, existing DM methods, constrained to Euclidean space, treat data as independent and identically distributed points, overlooking complex geometric and hierarchical relationships.
* **Signal Tags:** #ai

---


### Interpretable Machine Learning for Cognitive Aging: Handling Missing Data and Uncovering Social Determinant
**Date:** 2025-10-15 | **Arxiv:** [2510.10952](https://hub.bitwiki.org/t/interpretable-machine-learning-for-cognitive-aging-handling-missing-data-and-uncovering-social-determinant/16800)

#### Abstract
Early detection of Alzheimer's disease (AD) is crucial because its neurodegenerative effects are irreversible, and neuropathologic and social-behavioral risk factors accumulate years before diagnosis. Identifying higher-risk individuals earlier enables prevention, timely care, and equitable resource allocation. We predict cognitive performance from social determinants of health (SDOH) using the NIH NIA-supported PREPARE Challenge Phase 2 dataset derived from the nationally representative Mex-Cog cohort of the 2003 and 2012 Mexican Health and Aging Study (MHAS).   Data: The target is a validated composite cognitive score across seven domains-orientation, memory, attention, language, constructional praxis, and executive function-derived from the 2016 and 2021 MHAS waves. Predictors span demographic, socioeconomic, health, lifestyle, psychosocial, and healthcare access factors.   Methodology: Missingness was addressed with a singular value decomposition (SVD)-based imputation pipeline treating continuous and categorical variables separately. This approach leverages latent feature correlations to recover missing values while balancing reliability and scalability. After evaluating multiple methods, XGBoost was chosen for its superior predictive performance.   Results and Discussion: The framework outperformed existing methods and the data challenge leaderboard, demonstrating high accuracy, robustness, and interpretability. SHAP-based post hoc analysis identified top contributing SDOH factors and age-specific feature patterns. Notably, flooring material emerged as a strong predictor, reflecting socioeconomic and environmental disparities. Other influential factors, age, SES, lifestyle, social interaction, sleep, stress, and BMI, underscore the multifactorial nature of cognitive aging and the value of interpretable, data-driven SDOH modeling.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Homomorphic Mappings for Value-Preserving State Aggregation in Markov Decision Processes
**Date:** 2025-10-14 | **Arxiv:** [2510.09965](https://hub.bitwiki.org/t/homomorphic-mappings-for-value-preserving-state-aggregation-in-markov-decision-processes/16551)

#### Abstract
State aggregation aims to reduce the computational complexity of solving Markov Decision Processes (MDPs) while preserving the performance of the original system. A fundamental challenge lies in optimizing policies within the aggregated, or abstract, space such that the performance remains optimal in the ground MDP-a property referred to as {"}optimal policy equivalence {"}.   This paper presents an abstraction framework based on the notion of homomorphism, in which two Markov chains are deemed homomorphic if their value functions exhibit a linear relationship. Within this theoretical framework, we establish a sufficient condition for the equivalence of optimal policy.   We further examine scenarios where the sufficient condition is not met and derive an upper bound on the approximation error and a performance lower bound for the objective function under the ground MDP. We propose Homomorphic Policy Gradient (HPG), which guarantees optimal policy equivalence under sufficient conditions, and its extension, Error-Bounded HPG (EBHPG), which balances computational efficiency and the performance loss induced by aggregation. In the experiments, we validated the theoretical results and conducted comparative evaluations against seven algorithms.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Reinforcement Learning from Probabilistic Forecasts for Safe Decision-Making via Conditional Value-at-Risk Planning
**Date:** 2025-10-10 | **Arxiv:** [2510.08226](https://hub.bitwiki.org/t/reinforcement-learning-from-probabilistic-forecasts-for-safe-decision-making-via-conditional-value-at-risk-planning/15987)

#### Abstract
Sequential decisions in volatile, high-stakes settings require more than maximizing expected return; they require principled uncertainty management. This paper presents the Uncertainty-Aware Markov Decision Process (UAMDP), a unified framework that couples Bayesian forecasting, posterior-sampling reinforcement learning, and planning under a conditional value-at-risk (CVaR) constraint. In a closed loop, the agent updates its beliefs over latent dynamics, samples plausible futures via Thompson sampling, and optimizes policies subject to preset risk tolerances. We establish regret bounds that converge to the Bayes-optimal benchmark under standard regularity conditions. We evaluate UAMDP in two domains-high-frequency equity trading and retail inventory control-both marked by structural uncertainty and economic volatility. Relative to strong deep learning baselines, UAMDP improves long-horizon forecasting accuracy (RMSE decreases by up to 25\% and sMAPE by 32\%), and these gains translate into economic performance: the trading Sharpe ratio rises from 1.54 to 1.74 while maximum drawdown is roughly halved. These results show that integrating calibrated probabilistic modeling, exploration aligned with posterior uncertainty, and risk-aware control yields a robust, generalizable approach to safer and more profitable sequential decision-making.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Explaining Grokking and Information Bottleneck through Neural Collapse Emergence
**Date:** 2025-09-26 | **Arxiv:** [2509.20829](https://hub.bitwiki.org/t/explaining-grokking-and-information-bottleneck-through-neural-collapse-emergence/11658)

#### Abstract
The training dynamics of deep neural networks often defy expectations, even as these models form the foundation of modern machine learning. Two prominent examples are grokking, where test performance improves abruptly long after the training loss has plateaued, and the information bottleneck principle, where models progressively discard input information irrelevant to the prediction task as training proceeds. However, the mechanisms underlying these phenomena and their relations remain poorly understood. In this work, we present a unified explanation of such late-phase phenomena through the lens of neural collapse, which characterizes the geometry of learned representations. We show that the contraction of population within-class variance is a key factor underlying both grokking and information bottleneck, and relate this measure to the neural collapse measure defined on the training set. By analyzing the dynamics of neural collapse, we show that distinct time scales between fitting the training set and the progression of neural collapse account for the behavior of the late-phase phenomena. Finally, we validate our theoretical findings on multiple datasets and architectures.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, the mechanisms underlying these phenomena and their relations remain poorly understood.
* **Signal Tags:** #ai

---


### LogGuardQ: A Cognitive-Enhanced Reinforcement Learning Framework for Cybersecurity Anomaly Detection in Security Logs
**Date:** 2025-09-16 | **Arxiv:** [2509.10511](https://hub.bitwiki.org/t/logguardq-a-cognitive-enhanced-reinforcement-learning-framework-for-cybersecurity-anomaly-detection-in-security-logs/9391)

#### Abstract
Reinforcement learning (RL) has transformed sequential decision-making, but traditional algorithms like Deep Q-Networks (DQNs) and Proximal Policy Optimization (PPO) often struggle with efficient exploration, stability, and adaptability in dynamic environments. This study presents LogGuardQ (Adaptive Log Guard with Cognitive enhancement), a novel framework that integrates a dual-memory system inspired by human cognition and adaptive exploration strategies driven by temperature decay and curiosity. Evaluated on a dataset of 1,000,000 simulated access logs with 47.9% anomalies over 20,000 episodes, LogGuardQ achieves a 96.0% detection rate (versus 93.0% for DQN and 47.1% for PPO), with precision of 0.4776, recall of 0.9996, and an F1-score of 0.6450. The mean reward is 20.34 \pm 44.63 across all episodes (versus 18.80 \pm 43.98 for DQN and -0.17 \pm 23.79 for PPO), with an average of 5.0 steps per episode (constant across models). Graphical analyses, including learning curves smoothed with a Savgol filter (window=501, polynomial=2), variance trends, action distributions, and cumulative detections, demonstrate LogGuardQ's superior stability and efficiency. Statistical tests (Mann-Whitney U) confirm significant performance advantages (e.g., p = 0.0002 vs. DQN with negligible effect size, p < 0.0001 vs. PPO with medium effect size, and p < 0.0001 for DQN vs. PPO with small effect size). By bridging cognitive science and RL, LogGuardQ offers a scalable approach to adaptive learning in uncertain environments, with potential applications in cybersecurity, intrusion detection, and decision-making under uncertainty.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Towards Ontology-Based Descriptions of Conversations with Qualitatively-Defined Concepts
**Date:** 2025-09-08 | **Arxiv:** [2509.04926](https://hub.bitwiki.org/t/towards-ontology-based-descriptions-of-conversations-with-qualitatively-defined-concepts/8139)

#### Abstract
The controllability of Large Language Models (LLMs) when used as conversational agents is a key challenge, particularly to ensure predictable and user-personalized responses. This work proposes an ontology-based approach to formally define conversational features that are typically qualitative in nature. By leveraging a set of linguistic descriptors, we derive quantitative definitions for qualitatively-defined concepts, enabling their integration into an ontology for reasoning and consistency checking. We apply this framework to the task of proficiency-level control in conversations, using CEFR language proficiency levels as a case study. These definitions are then formalized in description logic and incorporated into an ontology, which guides controlled text generation of an LLM through fine-tuning. Experimental results demonstrate that our approach provides consistent and explainable proficiency-level definitions, improving transparency in conversational AI.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### SHeRL-FL: When Representation Learning Meets Split Learning in Hierarchical Federated Learning
**Date:** 2025-08-13 | **Arxiv:** [2508.08339](https://hub.bitwiki.org/t/sherl-fl-when-representation-learning-meets-split-learning-in-hierarchical-federated-learning/3233)

#### Abstract
Federated learning (FL) is a promising approach for addressing scalability and latency issues in large-scale networks by enabling collaborative model training without requiring the sharing of raw data. However, existing FL frameworks often overlook the computational heterogeneity of edge clients and the growing training burden on resource-limited devices. However, FL suffers from high communication costs and complex model aggregation, especially with large models. Previous works combine split learning (SL) and hierarchical FL (HierFL) to reduce device-side computation and improve scalability, but this introduces training complexity due to coordination across tiers. To address these issues, we propose SHeRL-FL, which integrates SL and hierarchical model aggregation and incorporates representation learning at intermediate layers. By allowing clients and edge servers to compute training objectives independently of the cloud, SHeRL-FL significantly reduces both coordination complexity and communication overhead. To evaluate the effectiveness and efficiency of SHeRL-FL, we performed experiments on image classification tasks using CIFAR-10, CIFAR-100, and HAM10000 with AlexNet, ResNet-18, and ResNet-50 in both IID and non-IID settings. In addition, we evaluate performance on image segmentation tasks using the ISIC-2018 dataset with a ResNet-50-based U-Net. Experimental results demonstrate that SHeRL-FL reduces data transmission by over 90\% compared to centralized FL and HierFL, and by 50\% compared to SplitFed, which is a hybrid of FL and SL, and further improves hierarchical split learning methods.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, existing FL frameworks often overlook the computational heterogeneity of edge clients and the growing training burden on resource-limited devices.
* **Signal Tags:** #ai

---


### Neural Surrogate HMC: On Using Neural Likelihoods for Hamiltonian Monte Carlo in Simulation-Based Inference
**Date:** 2025-12-10 | **Arxiv:** [2407.20432](https://hub.bitwiki.org/t/neural-surrogate-hmc-on-using-neural-likelihoods-for-hamiltonian-monte-carlo-in-simulation-based-inference/28571)

#### Abstract
Bayesian inference methods such as Markov Chain Monte Carlo (MCMC) typically require repeated computations of the likelihood function, but in some scenarios this is infeasible and alternative methods are needed. Simulation-based inference (SBI) methods address this problem by using machine learning to amortize computations. In this work, we highlight a particular synergy between the SBI method of neural likelihood estimation and the classic MCMC method of Hamiltonian Monte Carlo. We show that approximating the likelihood function with a neural network model can provide three distinct advantages: (1) amortizing the computations for MCMC; (2) providing gradients for Hamiltonian Monte Carlo, and (3) smoothing over noisy simulations resulting from numerical instabilities. We provide practical guidelines for defining a prior, sampling a training set, and evaluating convergence. The method is demonstrated in an application modeling the heliospheric transport of galactic cosmic rays, where it enables efficient inference of latent parameters in the Parker equation.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Handy Appetizer
**Date:** 2025-12-09 | **Arxiv:** [2409.17120](https://hub.bitwiki.org/t/deep-learning-and-machine-learning-advancing-big-data-analytics-and-management-handy-appetizer/28345)

#### Abstract
This book explores the role of Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) in driving the progress of big data analytics and management. The book focuses on simplifying the complex mathematical concepts behind deep learning, offering intuitive visualizations and practical case studies to help readers understand how neural networks and technologies like Convolutional Neural Networks (CNNs) work. It introduces several classic models and technologies such as Transformers, GPT, ResNet, BERT, and YOLO, highlighting their applications in fields like natural language processing, image recognition, and autonomous driving. The book also emphasizes the importance of pre-trained models and how they can enhance model performance and accuracy, with instructions on how to apply these models in various real-world scenarios. Additionally, it provides an overview of key big data management technologies like SQL and NoSQL databases, as well as distributed computing frameworks such as Apache Hadoop and Spark, explaining their importance in managing and processing vast amounts of data. Ultimately, the book underscores the value of mastering deep learning and big data management skills as critical tools for the future workforce, making it an essential resource for both beginners and experienced professionals.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### JEPA as a Neural Tokenizer: Learning Robust Speech Representations with Density Adaptive Attention
**Date:** 2025-12-09 | **Arxiv:** [2512.07168](https://hub.bitwiki.org/t/jepa-as-a-neural-tokenizer-learning-robust-speech-representations-with-density-adaptive-attention/28266)

#### Abstract
We introduce a two-stage self-supervised framework that combines the Joint-Embedding Predictive Architecture (JEPA) with a Density Adaptive Attention Mechanism (DAAM) for learning robust speech representations. Stage~1 uses JEPA with DAAM to learn semantic audio features via masked prediction in latent space, fully decoupled from waveform reconstruction. Stage~2 leverages these representations for efficient tokenization using Finite Scalar Quantization (FSQ) and a mixed-radix packing scheme, followed by high-fidelity waveform reconstruction with a HiFi-GAN decoder. By integrating Gaussian mixture-based density-adaptive gating into the JEPA encoder, the model performs adaptive temporal feature selection and discovers hierarchical speech structure at a low frame rate of 2.5~Hz. The resulting tokens (47.5 tokens/sec) provide a reversible, highly compressed, and language-model-friendly representation that is competitive with, and often more efficient than, existing neural audio codecs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Path Channels and Plan Extension Kernels: a Mechanistic Description of Planning in a Sokoban RNN
**Date:** 2025-12-05 | **Arxiv:** [2506.10138](https://hub.bitwiki.org/t/path-channels-and-plan-extension-kernels-a-mechanistic-description-of-planning-in-a-sokoban-rnn/27745)

#### Abstract
We partially reverse-engineer a convolutional recurrent neural network (RNN) trained with model-free reinforcement learning to play the box-pushing game Sokoban. We find that the RNN stores future moves (plans) as activations in particular channels of the hidden state, which we call path channels. A high activation in a particular location means that, when a box is in that location, it will get pushed in the channel's assigned direction. We examine the convolutional kernels between path channels and find that they encode the change in position resulting from each possible action, thus representing part of a learned transition model. The RNN constructs plans by starting at the boxes and goals. These kernels extend activations in path channels forwards from boxes and backwards from the goal. Negative values are placed in channels at obstacles. This causes the extension kernels to propagate the negative value in reverse, thus pruning the last few steps and letting an alternative plan emerge; a form of backtracking. Our work shows that, a precise understanding of the plan representation allows us to directly understand the bidirectional planning-like algorithm learned by model-free training in more familiar terms.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning to Clean: Reinforcement Learning for Noisy Label Correction
**Date:** 2025-11-26 | **Arxiv:** [2511.19808](https://hub.bitwiki.org/t/learning-to-clean-reinforcement-learning-for-noisy-label-correction/25854)

#### Abstract
The challenge of learning with noisy labels is significant in machine learning, as it can severely degrade the performance of prediction models if not addressed properly. This paper introduces a novel framework that conceptualizes noisy label correction as a reinforcement learning (RL) problem. The proposed approach, Reinforcement Learning for Noisy Label Correction (RLNLC), defines a comprehensive state space representing data and their associated labels, an action space that indicates possible label corrections, and a reward mechanism that evaluates the efficacy of label corrections. RLNLC learns a deep feature representation based policy network to perform label correction through reinforcement learning, utilizing an actor-critic method. The learned policy is subsequently deployed to iteratively correct noisy training labels and facilitate the training of the prediction model. The effectiveness of RLNLC is demonstrated through extensive experiments on multiple benchmark datasets, where it consistently outperforms existing state-of-the-art techniques for learning with noisy labels.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Reward Engineering for Spatial Epidemic Simulations: A Reinforcement Learning Platform for Individual Behavioral Learning
**Date:** 2025-11-25 | **Arxiv:** [2511.18000](https://hub.bitwiki.org/t/reward-engineering-for-spatial-epidemic-simulations-a-reinforcement-learning-platform-for-individual-behavioral-learning/25451)

#### Abstract
We present ContagionRL, a Gymnasium-compatible reinforcement learning platform specifically designed for systematic reward engineering in spatial epidemic simulations. Unlike traditional agent-based models that rely on fixed behavioral rules, our platform enables rigorous evaluation of how reward function design affects learned survival strategies across diverse epidemic scenarios. ContagionRL integrates a spatial SIRS+D epidemiological model with configurable environmental parameters, allowing researchers to stress-test reward functions under varying conditions including limited observability, different movement patterns, and heterogeneous population dynamics. We evaluate five distinct reward designs, ranging from sparse survival bonuses to a novel potential field approach, across multiple RL algorithms (PPO, SAC, A2C). Through systematic ablation studies, we identify that directional guidance and explicit adherence incentives are critical components for robust policy learning. Our comprehensive evaluation across varying infection rates, grid sizes, visibility constraints, and movement patterns reveals that reward function choice dramatically impacts agent behavior and survival outcomes. Agents trained with our potential field reward consistently achieve superior performance, learning maximal adherence to non-pharmaceutical interventions while developing sophisticated spatial avoidance strategies. The platform's modular design enables systematic exploration of reward-behavior relationships, addressing a knowledge gap in models of this type where reward engineering has received limited attention. ContagionRL is an effective platform for studying adaptive behavioral responses in epidemic contexts and highlight the importance of reward design, information structure, and environmental predictability in learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning to Admit Optimally in an $M/M/k/k+N$ Queueing System with Unknown Service Rate
**Date:** 2025-11-25 | **Arxiv:** [2202.02419](https://hub.bitwiki.org/t/learning-to-admit-optimally-in-an-m-m-k-k-n-queueing-system-with-unknown-service-rate/25734)

#### Abstract
Motivated by applications of the Erlang-B blocking model and the extended $M/M/k/k+N$ model that allows for some queueing, beyond communication networks to sizing and pricing in production, messaging, and app-based parking systems, we study admission control for such systems with unknown service rate. In our model, a dispatcher either admits every arrival into the system (when there is room) or blocks it. Every served job yields a fixed reward but incurs a per unit time holding cost which includes the waiting time in the queue to get service if there is any. We aim to design a dispatching policy that maximizes the long-term average reward by observing arrival times and system state at arrivals, a realistic decision-event driven sampling of such systems. The dispatcher observes neither service times nor departure epochs, which excludes the use of reward-based reinforcement learning approaches. We develop our learning-based dispatch scheme as a parametric learning problem a'la self-tuning adaptive control. In our problem, certainty equivalent control switches between always admit if room (explore infinitely often), and never admit (terminate learning), so at judiciously chosen times we avoid the never admit recommendation. We prove that our proposed policy asymptotically converges to the optimal policy and present finite-time regret guarantees. The extreme contrast in the control policies shows up in our regret bounds for different parameter regimes: constant in one versus logarithmic in another.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Bridging Philosophy and Machine Learning: A Structuralist Framework for Classifying Neural Network Representations
**Date:** 2025-11-25 | **Arxiv:** [2511.18633](https://hub.bitwiki.org/t/bridging-philosophy-and-machine-learning-a-structuralist-framework-for-classifying-neural-network-representations/25643)

#### Abstract
Machine learning models increasingly function as representational systems, yet the philosoph- ical assumptions underlying their internal structures remain largely unexamined. This paper develops a structuralist decision framework for classifying the implicit ontological commitments made in machine learning research on neural network representations. Using a modified PRISMA protocol, a systematic review of the last two decades of literature on representation learning and interpretability is conducted. Five influential papers are analysed through three hierarchical criteria derived from structuralist philosophy of science: entity elimination, source of structure, and mode of existence. The results reveal a pronounced tendency toward structural idealism, where learned representations are treated as model-dependent constructions shaped by architec- ture, data priors, and training dynamics. Eliminative and non-eliminative structuralist stances appear selectively, while structural realism is notably absent. The proposed framework clarifies conceptual tensions in debates on interpretability, emergence, and epistemic trust in machine learning, and offers a rigorous foundation for future interdisciplinary work between philosophy of science and machine learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### CLO: Efficient LLM Inference System with CPU-Light KVCache Offloading via Algorithm-System Co-Design
**Date:** 2025-11-19 | **Arxiv:** [2511.14510](https://hub.bitwiki.org/t/clo-efficient-llm-inference-system-with-cpu-light-kvcache-offloading-via-algorithm-system-co-design/24585)

#### Abstract
The growth of million-token LLMs exposes the scalability limits of inference systems, where the KVCache dominates memory usage and data transfer overhead. Recent offloading systems migrate the KVCache to CPU memory and incorporate top-k attention to reduce the volume of data transferred from the CPU, while further applying system-level optimizations such as on-GPU caching and prefetching to lower transfer overhead. However, they overlook the CPU bottleneck in three aspects: (1) substantial overhead of fine-grained dynamic cache management performed on the CPU side, (2) significant transfer overhead from poor PCIe bandwidth utilization caused by heavy gathering operations at the CPU side, and (3) GPU runtime bubbles introduced by coarse-grained CPU-centric synchronization. To address these challenges, we propose CLO, a CPU-light KVCache offloading system via algorithm-system co-design. CLO features: (1) a coarse-grained head-wise approximate on-GPU caching strategy with negligible cache management cost, (2) seamless combination of data prefetching and on-GPU persistent caching for lower transfer overhead, (3) a zero-copy transfer engine to fully exploit PCIe bandwidth, and a GPU-centric synchronization method to eliminate GPU stalls. Evaluation on two widely-used LLMs demonstrates that CLO achieves comparable accuracy to state-of-the-art systems, while substantially minimizing CPU overhead, fully utilizing PCIe bandwidth, thus improving decoding throughput by 9.3%-66.6%. Our results highlight that algorithm-system co-design is essential for memory-constrained LLM inference on modern GPU platforms. We open source CLO at https://github.com/CommediaJW/CLO.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, they overlook the CPU bottleneck in three aspects: (1) substantial overhead of fine-grained dynamic cache management performed on the CPU side, (2) significant transfer overhead from poor PCIe bandwidth utilization caused by heavy gathering operations at the CPU side, and (3) GPU runtime bubbles introduced by coarse-grained CPU-centric synchronization.
* **Signal Tags:** #ai

---


### Rethinking Saliency Maps: A Cognitive Human Aligned Taxonomy and Evaluation Framework for Explanations
**Date:** 2025-11-18 | **Arxiv:** [2511.13081](https://hub.bitwiki.org/t/rethinking-saliency-maps-a-cognitive-human-aligned-taxonomy-and-evaluation-framework-for-explanations/24348)

#### Abstract
Saliency maps are widely used for visual explanations in deep learning, but a fundamental lack of consensus persists regarding their intended purpose and alignment with diverse user queries. This ambiguity hinders the effective evaluation and practical utility of explanation methods. We address this gap by introducing the Reference-Frame $\times$ Granularity (RFxG) taxonomy, a principled conceptual framework that organizes saliency explanations along two essential axes:Reference-Frame: Distinguishing between pointwise ("Why this prediction?") and contrastive ("Why this and not an alternative?") explanations. Granularity: Ranging from fine-grained class-level (e.g., "Why Husky?") to coarse-grained group-level (e.g., "Why Dog?") interpretations. Using the RFxG lens, we demonstrate critical limitations in existing evaluation metrics, which overwhelmingly prioritize pointwise faithfulness while neglecting contrastive reasoning and semantic granularity. To systematically assess explanation quality across both RFxG dimensions, we propose four novel faithfulness metrics. Our comprehensive evaluation framework applies these metrics to ten state-of-the-art saliency methods, four model architectures, and three datasets. By advocating a shift toward user-intent-driven evaluation, our work provides both the conceptual foundation and the practical tools necessary to develop visual explanations that are not only faithful to the underlying model behavior but are also meaningfully aligned with the complexity of human understanding and inquiry.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Method of Manufactured Learning for Solver-free Training of Neural Operators
**Date:** 2025-11-18 | **Arxiv:** [2511.12890](https://hub.bitwiki.org/t/method-of-manufactured-learning-for-solver-free-training-of-neural-operators/24165)

#### Abstract
Training neural operators to approximate mappings between infinite-dimensional function spaces often requires extensive datasets generated by either demanding experimental setups or computationally expensive numerical solvers. This dependence on solver-based data limits scalability and constrains exploration across physical systems. Here we introduce the Method of Manufactured Learning (MML), a solver-independent framework for training neural operators using analytically constructed, physics-consistent datasets. Inspired by the classical method of manufactured solutions, MML replaces numerical data generation with functional synthesis, i.e., smooth candidate solutions are sampled from controlled analytical spaces, and the corresponding forcing fields are derived by direct application of the governing differential operators. During inference, setting these forcing terms to zero restores the original governing equations, allowing the trained neural operator to emulate the true solution operator of the system. The framework is agnostic to network architecture and can be integrated with any operator learning paradigm. In this paper, we employ Fourier neural operator as a representative example. Across canonical benchmarks including heat, advection, Burgers, and diffusion-reaction equations. MML achieves high spectral accuracy, low residual errors, and strong generalization to unseen conditions. By reframing data generation as a process of analytical synthesis, MML offers a scalable, solver-agnostic pathway toward constructing physically grounded neural operators that retain fidelity to governing laws without reliance on expensive numerical simulations or costly experimental data for training.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### BlinDNO: A Distributional Neural Operator for Dynamical System Reconstruction from Time-Label-Free data
**Date:** 2025-11-18 | **Arxiv:** [2511.12316](https://hub.bitwiki.org/t/blindno-a-distributional-neural-operator-for-dynamical-system-reconstruction-from-time-label-free-data/24105)

#### Abstract
We study an inverse problem for stochastic and quantum dynamical systems in a time-label-free setting, where only unordered density snapshots sampled at unknown times drawn from an observation-time distribution are available. These observations induce a distribution over state densities, from which we seek to recover the parameters of the underlying evolution operator. We formulate this as learning a distribution-to-function neural operator and propose BlinDNO, a permutation-invariant architecture that integrates a multiscale U-Net encoder with an attention-based mixer. Numerical experiments on a wide range of stochastic and quantum systems, including a 3D protein-folding mechanism reconstruction problem in a cryo-EM setting, demonstrate that BlinDNO reliably recovers governing parameters and consistently outperforms existing neural inverse operator baselines.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Quantile Q-Learning: Revisiting Offline Extreme Q-Learning with Quantile Regression
**Date:** 2025-11-18 | **Arxiv:** [2511.11973](https://hub.bitwiki.org/t/quantile-q-learning-revisiting-offline-extreme-q-learning-with-quantile-regression/24077)

#### Abstract
Offline reinforcement learning (RL) enables policy learning from fixed datasets without further environment interaction, making it particularly valuable in high-risk or costly domains. Extreme $Q$-Learning (XQL) is a recent offline RL method that models Bellman errors using the Extreme Value Theorem, yielding strong empirical performance. However, XQL and its stabilized variant MXQL suffer from notable limitations: both require extensive hyperparameter tuning specific to each dataset and domain, and also exhibit instability during training. To address these issues, we proposed a principled method to estimate the temperature coefficient $β$ via quantile regression under mild assumptions. To further improve training stability, we introduce a value regularization technique with mild generalization, inspired by recent advances in constrained value learning. Experimental results demonstrate that the proposed algorithm achieves competitive or superior performance across a range of benchmark tasks, including D4RL and NeoRL2, while maintaining stable training dynamics and using a consistent set of hyperparameters across all datasets and domains.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, XQL and its stabilized variant MXQL suffer from notable limitations: both require extensive hyperparameter tuning specific to each dataset and domain, and also exhibit instability during training.
* **Signal Tags:** #ai

---


### On the emergence of numerical instabilities in Next Generation Reservoir Computing
**Date:** 2025-11-18 | **Arxiv:** [2505.00846](https://hub.bitwiki.org/t/on-the-emergence-of-numerical-instabilities-in-next-generation-reservoir-computing/24019)

#### Abstract
Next Generation Reservoir Computing (NGRC) is a low-cost machine learning method for forecasting chaotic time series from data. Computational efficiency is crucial for scalable reservoir computing, requiring better strategies to reduce training cost. In this work, we uncover a connection between the numerical conditioning of the NGRC feature matrix -- formed by polynomial evaluations on time-delay coordinates -- and the long-term NGRC dynamics. We show that NGRC can be trained without regularization, reducing computational time. Our contributions are twofold. First, merging tools from numerical linear algebra and ergodic theory of dynamical systems, we systematically study how the feature matrix conditioning varies across hyperparameters. We demonstrate that the NGRC feature matrix tends to be ill-conditioned for short time lags, high-degree polynomials, and short length of training data. Second, we evaluate the impact of different numerical algorithms (Cholesky, singular value decomposition (SVD), and lower-upper (LU) decomposition) for solving the regularized least-squares problem. Our results reveal that SVD-based training achieves accurate forecasts without regularization, being preferable when compared against the other algorithms.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning
**Date:** 2025-11-12 | **Arxiv:** [2511.07730](https://hub.bitwiki.org/t/multistep-quasimetric-learning-for-scalable-goal-conditioned-reinforcement-learning/22960)

#### Abstract
Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Kolmogorov-Arnold Chemical Reaction Neural Networks for learning pressure-dependent kinetic rate laws
**Date:** 2025-11-12 | **Arxiv:** [2511.07686](https://hub.bitwiki.org/t/kolmogorov-arnold-chemical-reaction-neural-networks-for-learning-pressure-dependent-kinetic-rate-laws/23051)

#### Abstract
Chemical Reaction Neural Networks (CRNNs) have emerged as an interpretable machine learning framework for discovering reaction kinetics directly from data, while strictly adhering to the Arrhenius and mass action laws. However, standard CRNNs cannot represent pressure-dependent rate behavior, which is critical in many combustion and chemical systems and typically requires empirical formulations such as Troe or PLOG. Here, we develop Kolmogorov-Arnold Chemical Reaction Neural Networks (KA-CRNNs) that generalize CRNNs by modeling each kinetic parameter as a learnable function of system pressure using Kolmogorov-Arnold activations. This structure maintains full interpretability and physical consistency while enabling assumption-free inference of pressure effects directly from data. A proof-of-concept study on the CH3 recombination reaction demonstrates that KA-CRNNs accurately reproduce pressure-dependent kinetics across a range of temperatures and pressures, outperforming conventional interpolative models. The framework establishes a foundation for data-driven discovery of extended kinetic behaviors in complex reacting systems, advancing interpretable and physics-consistent approaches for chemical model inference.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, standard CRNNs cannot represent pressure-dependent rate behavior, which is critical in many combustion and chemical systems and typically requires empirical formulations such as Troe or PLOG.
* **Signal Tags:** #ai

---


### Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning
**Date:** 2025-11-12 | **Arxiv:** [2506.04411](https://hub.bitwiki.org/t/self-supervised-contrastive-learning-is-approximately-supervised-contrastive-learning/23128)

#### Abstract
Despite its empirical success, the theoretical foundations of self-supervised contrastive learning (CL) are not yet fully established. In this work, we address this gap by showing that standard CL objectives implicitly approximate a supervised variant we call the negatives-only supervised contrastive loss (NSCL), which excludes same-class contrasts. We prove that the gap between the CL and NSCL losses vanishes as the number of semantic classes increases, under a bound that is both label-agnostic and architecture-independent.   We characterize the geometric structure of the global minimizers of the NSCL loss: the learned representations exhibit augmentation collapse, within-class collapse, and class centers that form a simplex equiangular tight frame. We further introduce a new bound on the few-shot error of linear-probing. This bound depends on two measures of feature variability--within-class dispersion and variation along the line between class centers. We show that directional variation dominates the bound and that the within-class dispersion's effect diminishes as the number of labeled samples increases. These properties enable CL and NSCL-trained representations to support accurate few-shot label recovery using simple linear probes.   Finally, we empirically validate our theoretical findings: the gap between CL and NSCL losses decays at a rate of $\mathcal{O}(\frac{1}{\#\text{classes}})$; the two losses are highly correlated; minimizing the CL loss implicitly brings the NSCL loss close to the value achieved by direct minimization; and the proposed few-shot error bound provides a tight estimate of probing performance in practice. The code and project page of the paper are available at [\href{https://github.com/DLFundamentals/understanding-ssl}{code}, \href{https://dlfundamentals.github.io/ssl-is-approximately-sl/}{project page}].

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Understanding the role of depth in the neural tangent kernel for overparameterized neural networks
**Date:** 2025-11-11 | **Arxiv:** [2511.07272](https://hub.bitwiki.org/t/understanding-the-role-of-depth-in-the-neural-tangent-kernel-for-overparameterized-neural-networks/22506)

#### Abstract
Overparameterized fully-connected neural networks have been shown to behave like kernel models when trained with gradient descent, under mild conditions on the width, the learning rate, and the parameter initialization. In the limit of infinitely large widths and small learning rate, the kernel that is obtained allows to represent the output of the learned model with a closed-form solution. This closed-form solution hinges on the invertibility of the limiting kernel, a property that often holds on real-world datasets. In this work, we analyze the sensitivity of large ReLU networks to increasing depths by characterizing the corresponding limiting kernel. Our theoretical results demonstrate that the normalized limiting kernel approaches the matrix of ones. In contrast, they show the corresponding closed-form solution approaches a fixed limit on the sphere. We empirically evaluate the order of magnitude in network depth required to observe this convergent behavior, and we describe the essential properties that enable the generalization of our results to other kernels.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning Task Representations from In-Context Learning
**Date:** 2025-11-11 | **Arxiv:** [2502.05390](https://hub.bitwiki.org/t/learning-task-representations-from-in-context-learning/22888)

#### Abstract
Large language models (LLMs) have demonstrated remarkable proficiency in in-context learning (ICL), where models adapt to new tasks through example-based prompts without requiring parameter updates. However, understanding how tasks are internally encoded and generalized remains a challenge. To address some of the empirical and technical gaps in the literature, we introduce an automated formulation for encoding task information in ICL prompts as a function of attention heads within the transformer architecture. This approach computes a single task vector as a weighted sum of attention heads, with the weights optimized causally via gradient descent. Our findings show that existing methods fail to generalize effectively to modalities beyond text. In response, we also design a benchmark to evaluate whether a task vector can preserve task fidelity in functional regression tasks. The proposed method successfully extracts task-specific information from in-context demonstrations and excels in both text and regression tasks, demonstrating its generalizability across modalities.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, understanding how tasks are internally encoded and generalized remains a challenge.
* **Signal Tags:** #ai

---


### Risk Prediction of Cardiovascular Disease for Diabetic Patients with Machine Learning and Deep Learning Techniques
**Date:** 2025-11-10 | **Arxiv:** [2511.04971](https://hub.bitwiki.org/t/risk-prediction-of-cardiovascular-disease-for-diabetic-patients-with-machine-learning-and-deep-learning-techniques/22297)

#### Abstract
Accurate prediction of cardiovascular disease (CVD) risk is crucial for healthcare institutions. This study addresses the growing prevalence of diabetes and its strong link to heart disease by proposing an efficient CVD risk prediction model for diabetic patients using machine learning (ML) and hybrid deep learning (DL) approaches. The BRFSS dataset was preprocessed by removing duplicates, handling missing values, identifying categorical and numerical features, and applying Principal Component Analysis (PCA) for feature extraction. Several ML models, including Decision Trees (DT), Random Forest (RF), k-Nearest Neighbors (KNN), Support Vector Machine (SVM), AdaBoost, and XGBoost, were implemented, with XGBoost achieving the highest accuracy of 0.9050. Various DL models, such as Artificial Neural Networks (ANN), Deep Neural Networks (DNN), Recurrent Neural Networks (RNN), Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), Bidirectional LSTM (BiLSTM), and Gated Recurrent Unit (GRU), as well as hybrid models combining CNN with LSTM, BiLSTM, and GRU, were also explored. Some of these models achieved perfect recall (1.00), with the LSTM model achieving the highest accuracy of 0.9050. Our research highlights the effectiveness of ML and DL models in predicting CVD risk among diabetic patients, automating and enhancing clinical decision-making. High accuracy and F1 scores demonstrate these models' potential to improve personalized risk management and preventive strategies.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments
**Date:** 2025-11-06 | **Arxiv:** [2511.03527](https://hub.bitwiki.org/t/learning-without-critics-revisiting-grpo-in-classical-reinforcement-learning-environments/21850)

#### Abstract
Group Relative Policy Optimization (GRPO) has emerged as a scalable alternative to Proximal Policy Optimization (PPO) by eliminating the learned critic and instead estimating advantages through group-relative comparisons of trajectories. This simplification raises fundamental questions about the necessity of learned baselines in policy-gradient methods. We present the first systematic study of GRPO in classical single-task reinforcement learning environments, spanning discrete and continuous control tasks. Through controlled ablations isolating baselines, discounting, and group sampling, we reveal three key findings: (1) learned critics remain essential for long-horizon tasks: all critic-free baselines underperform PPO except in short-horizon environments like CartPole where episodic returns can be effective; (2) GRPO benefits from high discount factors (gamma = 0.99) except in HalfCheetah, where lack of early termination favors moderate discounting (gamma = 0.9); (3) smaller group sizes outperform larger ones, suggesting limitations in batch-based grouping strategies that mix unrelated episodes. These results reveal both the limitations of critic-free methods in classical control and the specific conditions where they remain viable alternatives to learned value functions.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning Under Laws: A Constraint-Projected Neural PDE Solver that Eliminates Hallucinations
**Date:** 2025-11-06 | **Arxiv:** [2511.03578](https://hub.bitwiki.org/t/learning-under-laws-a-constraint-projected-neural-pde-solver-that-eliminates-hallucinations/21856)

#### Abstract
Neural networks can approximate solutions to partial differential equations, but they often break the very laws they are meant to model-creating mass from nowhere, drifting shocks, or violating conservation and entropy. We address this by training within the laws of physics rather than beside them. Our framework, called Constraint-Projected Learning (CPL), keeps every update physically admissible by projecting network outputs onto the intersection of constraint sets defined by conservation, Rankine-Hugoniot balance, entropy, and positivity. The projection is differentiable and adds only about 10% computational overhead, making it fully compatible with back-propagation. We further stabilize training with total-variation damping (TVD) to suppress small oscillations and a rollout curriculum that enforces consistency over long prediction horizons. Together, these mechanisms eliminate both hard and soft violations: conservation holds at machine precision, total-variation growth vanishes, and entropy and error remain bounded. On Burgers and Euler systems, CPL produces stable, physically lawful solutions without loss of accuracy. Instead of hoping neural solvers will respect physics, CPL makes that behavior an intrinsic property of the learning process.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning
**Date:** 2025-11-05 | **Arxiv:** [2511.02567](https://hub.bitwiki.org/t/adaptive-neighborhood-constrained-q-learning-for-offline-reinforcement-learning/21601)

#### Abstract
Offline reinforcement learning (RL) suffers from extrapolation errors induced by out-of-distribution (OOD) actions. To address this, offline RL algorithms typically impose constraints on action selection, which can be systematically categorized into density, support, and sample constraints. However, we show that each category has inherent limitations: density and sample constraints tend to be overly conservative in many scenarios, while the support constraint, though least restrictive, faces challenges in accurately modeling the behavior policy. To overcome these limitations, we propose a new neighborhood constraint that restricts action selection in the Bellman target to the union of neighborhoods of dataset actions. Theoretically, the constraint not only bounds extrapolation errors and distribution shift under certain conditions, but also approximates the support constraint without requiring behavior policy modeling. Moreover, it retains substantial flexibility and enables pointwise conservatism by adapting the neighborhood radius for each data point. In practice, we employ data quality as the adaptation criterion and design an adaptive neighborhood constraint. Building on an efficient bilevel optimization framework, we develop a simple yet effective algorithm, Adaptive Neighborhood-constrained Q learning (ANQ), to perform Q learning with target actions satisfying this constraint. Empirically, ANQ achieves state-of-the-art performance on standard offline RL benchmarks and exhibits strong robustness in scenarios with noisy or limited data.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, we show that each category has inherent limitations: density and sample constraints tend to be overly conservative in many scenarios, while the support constraint, though least restrictive, faces challenges in accurately modeling the behavior policy.
* **Signal Tags:** #ai

---


### OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning
**Date:** 2025-11-05 | **Arxiv:** [2511.02205](https://hub.bitwiki.org/t/omnifield-conditioned-neural-fields-for-robust-multimodal-spatiotemporal-learning/21547)

#### Abstract
Multimodal spatiotemporal learning on real-world experimental data is constrained by two challenges: within-modality measurements are sparse, irregular, and noisy (QA/QC artifacts) but cross-modally correlated; the set of available modalities varies across space and time, shrinking the usable record unless models can adapt to arbitrary subsets at train and test time. We propose OmniField, a continuity-aware framework that learns a continuous neural field conditioned on available modalities and iteratively fuses cross-modal context. A multimodal crosstalk block architecture paired with iterative cross-modal refinement aligns signals prior to the decoder, enabling unified reconstruction, interpolation, forecasting, and cross-modal prediction without gridding or surrogate preprocessing. Extensive evaluations show that OmniField consistently outperforms eight strong multimodal spatiotemporal baselines. Under heavy simulated sensor noise, performance remains close to clean-input levels, highlighting robustness to corrupted measurements.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Path-Coordinated Continual Learning with Neural Tangent Kernel-Justified Plasticity: A Theoretical Framework with Near State-of-the-Art Performance
**Date:** 2025-11-05 | **Arxiv:** [2511.02025](https://hub.bitwiki.org/t/path-coordinated-continual-learning-with-neural-tangent-kernel-justified-plasticity-a-theoretical-framework-with-near-state-of-the-art-performance/21520)

#### Abstract
Catastrophic forgetting is one of the fundamental issues of continual learning because neural networks forget the tasks learned previously when trained on new tasks. The proposed framework is a new path-coordinated framework of continual learning that unites the Neural Tangent Kernel (NTK) theory of principled plasticity bounds, statistical validation by Wilson confidence intervals, and evaluation of path quality by the use of multiple metrics. Experimental evaluation shows an average accuracy of 66.7% at the cost of 23.4% catastrophic forgetting on Split-CIFAR10, a huge improvement over the baseline and competitive performance achieved, which is very close to state-of-the-art results. Further, it is found out that NTK condition numbers are predictive indicators of learning capacity limits, showing the existence of a critical threshold at condition number $>10^{11}$. It is interesting to note that the proposed strategy shows a tendency of lowering forgetting as the sequence of tasks progresses (27% to 18%), which is a system stabilization. The framework validates 80% of discovered paths with a rigorous statistical guarantee and maintains 90-97% retention on intermediate tasks. The core capacity limits of the continual learning environment are determined in the analysis, and actionable insights to enhance the adaptive regularization are offered.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Chronic Diseases Prediction using Machine Learning and Deep Learning Methods
**Date:** 2025-11-04 | **Arxiv:** [2505.00189](https://hub.bitwiki.org/t/chronic-diseases-prediction-using-machine-learning-and-deep-learning-methods/21396)

#### Abstract
Chronic diseases, such as cardiovascular disease, diabetes, chronic kidney disease, and thyroid disorders, are the leading causes of premature mortality worldwide. Early detection and intervention are crucial for improving patient outcomes, yet traditional diagnostic methods often fail due to the complex nature of these conditions. This study explores the application of machine learning (ML) and deep learning (DL) techniques to predict chronic disease and thyroid disorders. We used a variety of models, including Logistic Regression (LR), Random Forest (RF), Gradient Boosted Trees (GBT), Neural Networks (NN), Decision Trees (DT) and Native Bayes (NB), to analyze and predict disease outcomes. Our methodology involved comprehensive data pre-processing, including handling missing values, categorical encoding, and feature aggregation, followed by model training and evaluation. Performance metrics such ad precision, recall, accuracy, F1-score, and Area Under the Curve (AUC) were used to assess the effectiveness of each model. The results demonstrated that ensemble methods like Random Forest and Gradient Boosted Trees consistently outperformed. Neutral Networks also showed superior performance, particularly in capturing complex data patterns. The findings highlight the potential of ML and DL in revolutionizing chronic disease prediction, enabling early diagnosis and personalized treatment strategies. However, challenges such as data quality, model interpretability, and the need for advanced computational techniques in healthcare to improve patient outcomes and reduce the burden of chronic diseases. This study was conducted as part of Big Data class project under the supervision of our professors Mr. Abderrahmane EZ-ZAHOUT and Mr. Abdessamad ESSAIDI.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, challenges such as data quality, model interpretability, and the need for advanced computational techniques in healthcare to improve patient outcomes and reduce the burden of chronic diseases.
* **Signal Tags:** #ai

---


### Learning what to say and how precisely: Efficient Communication via Differentiable Discrete Communication Learning
**Date:** 2025-11-04 | **Arxiv:** [2511.01554](https://hub.bitwiki.org/t/learning-what-to-say-and-how-precisely-efficient-communication-via-differentiable-discrete-communication-learning/21337)

#### Abstract
Effective communication in multi-agent reinforcement learning (MARL) is critical for success but constrained by bandwidth, yet past approaches have been limited to complex gating mechanisms that only decide \textit{whether} to communicate, not \textit{how precisely}. Learning to optimize message precision at the bit-level is fundamentally harder, as the required discretization step breaks gradient flow. We address this by generalizing Differentiable Discrete Communication Learning (DDCL), a framework for end-to-end optimization of discrete messages. Our primary contribution is an extension of DDCL to support unbounded signals, transforming it into a universal, plug-and-play layer for any MARL architecture. We verify our approach with three key results. First, through a qualitative analysis in a controlled environment, we demonstrate \textit{how} agents learn to dynamically modulate message precision according to the informational needs of the task. Second, we integrate our variant of DDCL into four state-of-the-art MARL algorithms, showing it reduces bandwidth by over an order of magnitude while matching or exceeding task performance. Finally, we provide direct evidence for the \enquote{Bitter Lesson} in MARL communication: a simple Transformer-based policy leveraging DDCL matches the performance of complex, specialized architectures, questioning the necessity of bespoke communication designs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### SmartMixed: A Two-Phase Training Strategy for Adaptive Activation Function Learning in Neural Networks
**Date:** 2025-11-03 | **Arxiv:** [2510.22450](https://hub.bitwiki.org/t/smartmixed-a-two-phase-training-strategy-for-adaptive-activation-function-learning-in-neural-networks/20962)

#### Abstract
The choice of activation function plays a critical role in neural networks, yet most architectures still rely on fixed, uniform activation functions across all neurons. We introduce SmartMixed, a two-phase training strategy that allows networks to learn optimal per-neuron activation functions while preserving computational efficiency at inference. In the first phase, neurons adaptively select from a pool of candidate activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, SELU) using a differentiable hard-mixture mechanism. In the second phase, each neuron's activation function is fixed according to the learned selection, resulting in a computationally efficient network that supports continued training with optimized vectorized operations. We evaluate SmartMixed on the MNIST dataset using feedforward neural networks of varying depths. The analysis shows that neurons in different layers exhibit distinct preferences for activation functions, providing insights into the functional diversity within neural architectures.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning to Insert for Constructive Neural Vehicle Routing Solver
**Date:** 2025-10-31 | **Arxiv:** [2505.13904](https://hub.bitwiki.org/t/learning-to-insert-for-constructive-neural-vehicle-routing-solver/20693)

#### Abstract
Neural Combinatorial Optimisation (NCO) is a promising learning-based approach for solving Vehicle Routing Problems (VRPs) without extensive manual design. While existing constructive NCO methods typically follow an appending-based paradigm that sequentially adds unvisited nodes to partial solutions, this rigid approach often leads to suboptimal results. To overcome this limitation, we explore the idea of insertion-based paradigm and propose Learning to Construct with Insertion-based Paradigm (L2C-Insert), a novel learning-based method for constructive NCO. Unlike traditional approaches, L2C-Insert builds solutions by strategically inserting unvisited nodes at any valid position in the current partial solution, which can significantly enhance the flexibility and solution quality. The proposed framework introduces three key components: a novel model architecture for precise insertion position prediction, an efficient training scheme for model optimization, and an advanced inference technique that fully exploits the insertion paradigm's flexibility. Extensive experiments on both synthetic and real-world instances of the Travelling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that L2C-Insert consistently achieves superior performance across various problem sizes.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
