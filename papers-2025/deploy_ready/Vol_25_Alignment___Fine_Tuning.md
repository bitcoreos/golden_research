# Vol 25 Alignment   Fine Tuning
*Enriched by BITCOREOS | Phase 4 Batch 5*

---

### VLMLight: Safety-Critical Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning Architecture
**Date:** 2025-10-20 | **Arxiv:** [2505.19486](https://hub.bitwiki.org/t/vlmlight-safety-critical-traffic-signal-control-via-vision-language-meta-control-and-dual-branch-reasoning-architecture/17914)

#### Abstract
Traffic signal control (TSC) is a core challenge in urban mobility, where real-time decisions must balance efficiency and safety. Existing methods - ranging from rule-based heuristics to reinforcement learning (RL) - often struggle to generalize to complex, dynamic, and safety-critical scenarios. We introduce VLMLight, a novel TSC framework that integrates vision-language meta-control with dual-branch reasoning. At the core of VLMLight is the first image-based traffic simulator that enables multi-view visual perception at intersections, allowing policies to reason over rich cues such as vehicle type, motion, and spatial density. A large language model (LLM) serves as a safety-prioritized meta-controller, selecting between a fast RL policy for routine traffic and a structured reasoning branch for critical cases. In the latter, multiple LLM agents collaborate to assess traffic phases, prioritize emergency vehicles, and verify rule compliance. Experiments show that VLMLight reduces waiting times for emergency vehicles by up to 65% over RL-only systems, while preserving real-time performance in standard conditions with less than 1% degradation. VLMLight offers a scalable, interpretable, and safety-aware solution for next-generation traffic signal control.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Regularization Through Reasoning: Systematic Improvements in Language Model Classification via Explanation-Enhanced Fine-Tuning
**Date:** 2025-11-05 | **Arxiv:** [2511.02044](https://hub.bitwiki.org/t/regularization-through-reasoning-systematic-improvements-in-language-model-classification-via-explanation-enhanced-fine-tuning/21525)

#### Abstract
Fine-tuning LLMs for classification typically maps inputs directly to labels. We ask whether attaching brief explanations to each label during fine-tuning yields better models. We evaluate conversational response quality along three axes: naturalness, comprehensiveness, and on-topic adherence, each rated on 5-point scales. Using ensemble-generated data from multiple LLMs, we fine-tune a 7B-parameter model and test across six diverse conversational datasets. Across 18 dataset, task settings, label-plus-explanation training outperforms label-only baselines.   A central and unexpected result concerns random tokens. We replace human-written explanations with text that is syntactically incoherent yet vocabulary-aligned with the originals (e.g., shuffled or bag-of-words variants). Despite lacking semantics, these pseudo-explanations still improve accuracy over label-only training and often narrow much of the gap to true explanations. The effect persists across datasets and training seeds, indicating that gains arise less from meaning than from structure: the extra token budget encourages richer intermediate computation and acts as a regularizer that reduces over-confident shortcuts.   Internal analyses support this view: explanation-augmented models exhibit higher activation entropy in intermediate layers alongside sharper predictive mass at the output layer, consistent with increased deliberation before decision. Overall, explanation-augmented fine-tuning, whether with genuine rationales or carefully constructed random token sequences, improves accuracy and reliability for LLM classification while clarifying how token-level scaffolding shapes computation during inference.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Do What You Say: Steering Vision-Language-Action Models via Runtime Reasoning-Action Alignment Verification
**Date:** 2025-10-21 | **Arxiv:** [2510.16281](https://hub.bitwiki.org/t/do-what-you-say-steering-vision-language-action-models-via-runtime-reasoning-action-alignment-verification/18268)

#### Abstract
Reasoning Vision Language Action (VLA) models improve robotic instruction-following by generating step-by-step textual plans before low-level actions, an approach inspired by Chain-of-Thought (CoT) reasoning in language models. Yet even with a correct textual plan, the generated actions can still miss the intended outcomes in the plan, especially in out-of-distribution (OOD) scenarios. We formalize this phenomenon as a lack of embodied CoT faithfulness, and introduce a training-free, runtime policy steering method for reasoning-action alignment. Given a reasoning VLA's intermediate textual plan, our framework samples multiple candidate action sequences from the same model, predicts their outcomes via simulation, and uses a pre-trained Vision-Language Model (VLM) to select the sequence whose outcome best aligns with the VLA's own textual plan. Only executing action sequences that align with the textual reasoning turns our base VLA's natural action diversity from a source of error into a strength, boosting robustness to semantic and visual OOD perturbations and enabling novel behavior composition without costly re-training. We also contribute a reasoning-annotated extension of LIBERO-100, environment variations tailored for OOD evaluation, and demonstrate up to 15% performance gain over prior work on behavior composition tasks and scales with compute and data diversity. Project Website at: https://yilin-wu98.github.io/steering-reasoning-vla/

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### ReasonIF: Large Reasoning Models Fail to Follow Instructions During Reasoning
**Date:** 2025-10-20 | **Arxiv:** [2510.15211](https://hub.bitwiki.org/t/reasonif-large-reasoning-models-fail-to-follow-instructions-during-reasoning/17742)

#### Abstract
The ability of large language models (LLMs) to follow user instructions is central to their reliability, safety, and usefulness. While prior studies assess instruction adherence in the model's main responses, we argue that it is also critical for large reasoning models (LRMs) to follow user instructions throughout their reasoning process. Reasoning instruction following makes LRMs more controllable and transparent, while reducing risks of undesirable shortcuts, hallucinations, or reward hacking within reasoning traces. To evaluate this dimension, we introduce ReasonIF, a systematic benchmark for assessing reasoning instruction following. ReasonIF includes six categories of instruction prompts, spanning multilingual reasoning, formatting and length control. Across many open-source LRMs including GPT-OSS, Qwen3, and DeepSeek-R1, we find substantial failures in reasoning instruction adherence: the highest instruction following score (IFS) remains below 0.25, meaning that fewer than $25\%$ of reasoning traces comply with the given instructions. Notably, as task difficulty increases, reasoning instruction following degrades further. We also explore two strategies to enhance reasoning instruction fidelity. (1) multi-turn reasoning and (2) Reasoning Instruction Finetuning (RIF) using synthetic data. RIF improves the IFS of $GPT-OSS-20B$ from 0.11 to 0.27, indicating measurable progress but leaving ample room for improvement.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer
**Date:** 2025-09-30 | **Arxiv:** [2502.00639](https://hub.bitwiki.org/t/half-order-fine-tuning-for-diffusion-model-a-recursive-likelihood-ratio-optimizer/12582)

#### Abstract
The probabilistic diffusion model (DM), generating content by inferencing through a recursive chain structure, has emerged as a powerful framework for visual generation. After pre-training on enormous data, the model needs to be properly aligned to meet requirements for downstream applications. How to efficiently align the foundation DM is a crucial task. Contemporary methods are either based on Reinforcement Learning (RL) or truncated Backpropagation (BP). However, RL and truncated BP suffer from low sample efficiency and biased gradient estimation, respectively, resulting in limited improvement or, even worse, complete training failure. To overcome the challenges, we propose the Recursive Likelihood Ratio (RLR) optimizer, a Half-Order (HO) fine-tuning paradigm for DM. The HO gradient estimator enables the computation graph rearrangement within the recursive diffusive chain, making the RLR's gradient estimator an unbiased one with lower variance than other methods. We theoretically investigate the bias, variance, and convergence of our method. Extensive experiments are conducted on image and video generation to validate the superiority of the RLR. Furthermore, we propose a novel prompt technique that is natural for the RLR to achieve a synergistic effect.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, RL and truncated BP suffer from low sample efficiency and biased gradient estimation, respectively, resulting in limited improvement or, even worse, complete training failure.
* **Signal Tags:** #ai

---


### Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning
**Date:** 2025-11-17 | **Arxiv:** [2505.16270](https://hub.bitwiki.org/t/transformer-copilot-learning-from-the-mistake-log-in-llm-fine-tuning/23874)

#### Abstract
Large language models are typically adapted to downstream tasks through supervised fine-tuning on domain-specific data. While standard fine-tuning focuses on minimizing generation loss to optimize model parameters, we take a deeper step by retaining and leveraging the model's own learning signals, analogous to how human learners reflect on past mistakes to improve future performance. We first introduce the concept of Mistake Log to systematically track the model's learning behavior and recurring errors throughout fine-tuning. Treating the original transformer-based model as the Pilot, we correspondingly design a Copilot model to refine the Pilot's inference performance via logits rectification. We name the overall Pilot-Copilot framework the Transformer Copilot, which introduces (i) a novel Copilot model design, (ii) a joint training paradigm where the Copilot continuously learns from the evolving Mistake Log alongside the Pilot, and (iii) a fused inference paradigm where the Copilot rectifies the Pilot's logits for enhanced generation. We provide both theoretical and empirical analyses on our new learning framework. Experiments on 12 benchmarks spanning commonsense, arithmetic, and recommendation tasks demonstrate that Transformer Copilot consistently improves performance by up to 34.5%, while introducing marginal computational overhead to Pilot models and exhibiting strong scalability and transferability. Our code is released at https://github.com/jiaruzouu/TransformerCopilot.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models
**Date:** 2025-10-22 | **Arxiv:** [2510.18526](https://hub.bitwiki.org/t/counterfactual-reasoning-for-steerable-pluralistic-value-alignment-of-large-language-models/18701)

#### Abstract
As large language models (LLMs) become increasingly integrated into applications serving users across diverse cultures, communities and demographics, it is critical to align LLMs with pluralistic human values beyond average principles (e.g., HHH). In psychological and social value theories such as Schwartz's Value Theory, pluralistic values are represented by multiple value dimensions paired with various priorities. However, existing methods encounter two challenges when aligning with such fine-grained value objectives: 1) they often treat multiple values as independent and equally important, ignoring their interdependence and relative priorities (value complexity); 2) they struggle to precisely control nuanced value priorities, especially those underrepresented ones (value steerability). To handle these challenges, we propose COUPLE, a COUnterfactual reasoning framework for PLuralistic valuE alignment. It introduces a structural causal model (SCM) to feature complex interdependency and prioritization among features, as well as the causal relationship between high-level value dimensions and behaviors. Moreover, it applies counterfactual reasoning to generate outputs aligned with any desired value objectives. Benefitting from explicit causal modeling, COUPLE also provides better interpretability. We evaluate COUPLE on two datasets with different value systems and demonstrate that COUPLE advances other baselines across diverse types of value objectives.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, existing methods encounter two challenges when aligning with such fine-grained value objectives: 1) they often treat multiple values as independent and equally important, ignoring their interdependence and relative priorities (value complexity); 2) they struggle to precisely control nuanced value priorities, especially those underrepresented ones (value steerability).
* **Signal Tags:** #ai

---


### Learning from Reference Answers: Versatile Language Model Alignment without Binary Human Preference Data
**Date:** 2025-10-15 | **Arxiv:** [2504.09895](https://hub.bitwiki.org/t/learning-from-reference-answers-versatile-language-model-alignment-without-binary-human-preference-data/17083)

#### Abstract
Large language models~(LLMs) are expected to be helpful, harmless, and honest. In different alignment scenarios, such as safety, confidence, and general preference alignment, binary preference data collection and reward modeling are resource-intensive but play a central role in transferring human preferences. In this work, we explore using the similarity between sampled generations and reference answers as a supplementary reward function for alignment. When unary reference answers are available, such similarity-based rewards can circumvent the need for binary preference data and explicit reward modeling. We introduce \textit{RefAlign}, a versatile REINFORCE-style alignment algorithm that does not rely on reward or reference models. RefAlign utilizes language generation evaluation metrics, such as BERTScore, between sampled generations and reference answers as surrogate rewards. Beyond general preference optimization, RefAlign can be naturally extended to diverse scenarios, including safety and confidence alignment, by combining similarity-based rewards with task-specific objectives. Across multiple scenarios, RefAlign achieves performance comparable to prior alignment methods while operating without binary preference data or reward models. The code is available at https://github.com/mzhaoshuai/RefAlign.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Model
**Date:** 2025-10-15 | **Arxiv:** [2510.10921](https://hub.bitwiki.org/t/fg-clip-2-a-bilingual-fine-grained-vision-language-alignment-model/16931)

#### Abstract
Fine-grained vision-language understanding requires precise alignment between visual content and linguistic descriptions, a capability that remains limited in current models, particularly in non-English settings. While models like CLIP perform well on global alignment, they often struggle to capture fine-grained details in object attributes, spatial relations, and linguistic expressions, with limited support for bilingual comprehension. To address these challenges, we introduce FG-CLIP 2, a bilingual vision-language model designed to advance fine-grained alignment for both English and Chinese. Our approach leverages rich fine-grained supervision, including region-text matching and long-caption modeling, alongside multiple discriminative objectives. We further introduce the Textual Intra-modal Contrastive (TIC) loss to better distinguish semantically similar captions. Trained on a carefully curated mixture of large-scale English and Chinese data, FG-CLIP 2 achieves powerful bilingual performance. To enable rigorous evaluation, we present a new benchmark for Chinese multimodal understanding, featuring long-caption retrieval and bounding box classification. Extensive experiments on 29 datasets across 8 tasks show that FG-CLIP 2 outperforms existing methods, achieving state-of-the-art results in both languages. We release the model, code, and benchmark to facilitate future research on bilingual fine-grained alignment.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Meta-Awareness Enhances Reasoning Models: Self-Alignment Reinforcement Learning
**Date:** 2025-10-07 | **Arxiv:** [2510.03259](https://hub.bitwiki.org/t/meta-awareness-enhances-reasoning-models-self-alignment-reinforcement-learning/14672)

#### Abstract
Recent studies on reasoning models explore the meta-awareness of language models, the ability to know how to think by itself. We argue that large reasoning models lack this meta-awareness property by proving severe misalignment between true rollouts and predicted meta information. We posit that aligning meta-prediction with true rollouts will lead to significant performance gains. To verify this hypothesis, we design a training pipeline that boosts Meta-Awareness via Self-Alignment (MASA), and prove that enhanced meta-awareness directly translates to improved accuracy. Unlike existing meta-cognitive reasoning models, our method does not require external training sources but leverages self-generated signals to train meta-awareness. Moreover, our method enables efficient training by i) filtering out zero-variance prompts that are either trivial or unsolvable and ii) cutting off lengthy rollouts when they are unlikely to lead to correct answers. The results are inspiring: our strategy yields significant improvements in both accuracy and training efficiency on in-domain tasks and shows strong generalization to out-of-domain benchmarks. More specifically, our method can speed up GRPO training by over 1.28x to reach the same performance, and achieve a 19.3% gain in accuracy on AIME25, and a 6.2 % average gain over six mathematics benchmarks. Training with meta-cognitive guidance enhances out-of-domain generalization, giving a 3.87 % boost on GPQA-Diamond and a 2.08 % overall accuracy gain across 13 benchmarks spanning logical, scientific, and coding domains.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### From Supervision to Exploration: What Does Protein Language Model Learn During Reinforcement Learning?
**Date:** 2025-10-03 | **Arxiv:** [2510.01571](https://hub.bitwiki.org/t/from-supervision-to-exploration-what-does-protein-language-model-learn-during-reinforcement-learning/14094)

#### Abstract
Protein language models (PLMs) have advanced computational protein science through large-scale pretraining and scalable architectures. In parallel, reinforcement learning (RL) has broadened exploration and enabled precise multi-objective optimization in protein design. Yet whether RL can push PLMs beyond their pretraining priors to uncover latent sequence-structure-function rules remains unclear. We address this by pairing RL with PLMs across four domains: antimicrobial peptide design, kinase variant optimization, antibody engineering, and inverse folding. Using diverse RL algorithms and model classes, we ask if RL improves sampling efficiency and, more importantly, if it reveals capabilities not captured by supervised learning. Across benchmarks, RL consistently boosts success rates and sample efficiency. Performance follows a three-factor interaction: task headroom, reward fidelity, and policy capacity jointly determine gains. When rewards are accurate and informative, policies have sufficient capacity, and tasks leave room beyond supervised baselines, improvements scale; when rewards are noisy or capacity is constrained, gains saturate despite exploration. This view yields practical guidance for RL in protein design: prioritize reward modeling and calibration before scaling policy size, match algorithm and regularization strength to task difficulty, and allocate capacity where marginal gains are largest. Implementation is available at https://github.com/chq1155/RL-PLM.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### LogicTree: Structured Proof Exploration for Coherent and Rigorous Logical Reasoning with Large Language Models
**Date:** 2025-09-16 | **Arxiv:** [2504.14089](https://hub.bitwiki.org/t/logictree-structured-proof-exploration-for-coherent-and-rigorous-logical-reasoning-with-large-language-models/9730)

#### Abstract
Large language models (LLMs) have achieved remarkable multi-step reasoning capabilities across various domains. However, LLMs still face distinct challenges in complex logical reasoning, as (1) proof-finding requires systematic exploration and the maintenance of logical coherence and (2) searching the right combination of premises at each reasoning step is inherently challenging in tasks with large premise space. To address this, we propose LogicTree, an inference-time modular framework employing algorithm-guided search to automate structured proof exploration and ensure logical coherence. Advancing beyond tree-of-thought (ToT), we incorporate caching mechanism into LogicTree to enable effective utilization of historical knowledge, preventing reasoning stagnation and minimizing redundancy. Furthermore, we address the combinatorial complexity of premise search by decomposing it into a linear process. The refined premise selection restricts subsequent inference to at most one derivation per step, enhancing reasoning granularity and enforcing strict step-by-step reasoning. Additionally, we introduce two LLM-free heuristics for premise prioritization, enabling strategic proof search. Experimental results on five datasets demonstrate that LogicTree optimally scales inference-time computation to achieve higher proof accuracy, surpassing chain-of-thought (CoT) and ToT with average gains of 23.6% and 12.5%, respectively, on GPT-4o. Moreover, within LogicTree, GPT-4o outperforms o3-mini by 7.6% on average.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, LLMs still face distinct challenges in complex logical reasoning, as (1) proof-finding requires systematic exploration and the maintenance of logical coherence and (2) searching the right combination of premises at each reasoning step is inherently challenging in tasks with large premise space.
* **Signal Tags:** #ai

---


### Evolutionary Architecture Search through Grammar-Based Sequence Alignment
**Date:** 2025-12-05 | **Arxiv:** [2512.04992](https://hub.bitwiki.org/t/evolutionary-architecture-search-through-grammar-based-sequence-alignment/27716)

#### Abstract
Neural architecture search (NAS) in expressive search spaces is a computationally hard problem, but it also holds the potential to automatically discover completely novel and performant architectures. To achieve this we need effective search algorithms that can identify powerful components and reuse them in new candidate architectures. In this paper, we introduce two adapted variants of the Smith-Waterman algorithm for local sequence alignment and use them to compute the edit distance in a grammar-based evolutionary architecture search. These algorithms enable us to efficiently calculate a distance metric for neural architectures and to generate a set of hybrid offspring from two parent models. This facilitates the deployment of crossover-based search heuristics, allows us to perform a thorough analysis on the architectural loss landscape, and track population diversity during search. We highlight how our method vastly improves computational complexity over previous work and enables us to efficiently compute shortest paths between architectures. When instantiating the crossover in evolutionary searches, we achieve competitive results, outperforming competing methods. Future work can build upon this new tool, discovering novel components that can be used more broadly across neural architecture design, and broadening its applications beyond NAS.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Fairness-Aware Fine-Tuning of Vision-Language Models for Medical Glaucoma Diagnosis
**Date:** 2025-12-04 | **Arxiv:** [2512.03477](https://hub.bitwiki.org/t/fairness-aware-fine-tuning-of-vision-language-models-for-medical-glaucoma-diagnosis/27528)

#### Abstract
Vision-language models achieve expert-level performance on medical imaging tasks but exhibit significant diagnostic accuracy disparities across demographic groups. We introduce fairness-aware Low-Rank Adaptation for medical VLMs, combining parameter efficiency with explicit fairness optimization. Our key algorithmic contribution is a differentiable MaxAccGap loss that enables end-to-end optimization of accuracy parity across demographic groups. We propose three methods: FR-LoRA integrates MaxAccGap regularization into the training objective, GR-LoRA applies inverse frequency weighting to balance gradient contributions, and Hybrid-LoRA combines both mechanisms. Evaluated on 10,000 glaucoma fundus images, GR-LoRA reduces diagnostic accuracy disparities by 69% while maintaining 53.15% overall accuracy. Ablation studies reveal that strong regularization strength achieves optimal fairness with minimal accuracy trade-off, and race-specific optimization yields 60% disparity reduction. Our approach requires only 0.24% trainable parameters, enabling practical deployment of fair medical AI in resource-constrained healthcare settings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Fine-Tuning Vision-Language Models for Multimodal Polymer Property Prediction
**Date:** 2025-11-11 | **Arxiv:** [2511.05577](https://hub.bitwiki.org/t/fine-tuning-vision-language-models-for-multimodal-polymer-property-prediction/22447)

#### Abstract
Vision-Language Models (VLMs) have shown strong performance in tasks like visual question answering and multimodal text generation, but their effectiveness in scientific domains such as materials science remains limited. While some machine learning methods have addressed specific challenges in this field, there is still a lack of foundation models designed for broad tasks like polymer property prediction using multimodal data. In this work, we present a multimodal polymer dataset to fine-tune VLMs through instruction-tuning pairs and assess the impact of multimodality on prediction performance. Our fine-tuned models, using LoRA, outperform unimodal and baseline approaches, demonstrating the benefits of multimodal learning. Additionally, this approach reduces the need to train separate models for different properties, lowering deployment and maintenance costs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Approximate non-linear model predictive control with safety-augmented neural networks
**Date:** 2025-11-07 | **Arxiv:** [2304.09575](https://hub.bitwiki.org/t/approximate-non-linear-model-predictive-control-with-safety-augmented-neural-networks/22161)

#### Abstract
Model predictive control (MPC) achieves stability and constraint satisfaction for general nonlinear systems, but requires computationally expensive online optimization. This paper studies approximations of such MPC controllers via neural networks (NNs) to achieve fast online evaluation. We propose safety augmentation that yields deterministic guarantees for convergence and constraint satisfaction despite approximation inaccuracies. We approximate the entire input sequence of the MPC with NNs, which allows us to verify online if it is a feasible solution to the MPC problem. We replace the NN solution by a safe candidate based on standard MPC techniques whenever it is infeasible or has worse cost. Our method requires a single evaluation of the NN and forward integration of the input sequence online, which is fast to compute on resource-constrained systems. The proposed control framework is illustrated using two numerical non-linear MPC benchmarks of different complexity, demonstrating computational speedups that are orders of magnitude higher than online optimization. In the examples, we achieve deterministic safety through the safety-augmented NNs, where a naive NN implementation fails.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### $\pi_\texttt{RL}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models
**Date:** 2025-10-31 | **Arxiv:** [2510.25889](https://hub.bitwiki.org/t/pi-texttt-rl-online-rl-fine-tuning-for-flow-based-vision-language-action-models/20500)

#### Abstract
Vision-Language-Action (VLA) models enable robots to understand and perform complex tasks from multimodal input. Although recent work explores using reinforcement learning (RL) to automate the laborious data collection process in scaling supervised fine-tuning (SFT), applying large-scale RL to flow-based VLAs (\eg, $π_0$, $π_{0.5}$) remains challenging due to intractable action log-likelihoods from iterative denoising. We address this challenge with $π_{\texttt{RL}}$, an open-source framework for training flow-based VLAs in parallel simulation. $π_{\texttt{RL}}$ implements two RL algorithms: (1) \textbf{Flow-Noise} models the denoising process as a discrete-time MDP with a learnable noise network for exact log-likelihood computation. (2) \textbf{Flow-SDE} integrates denoising with agent-environment interaction, formulating a two-layer MDP that employs ODE-to-SDE conversion for efficient RL exploration. We evaluate $π_{\texttt{RL}}$ on LIBERO, ManiSkill, and MetaWorld benchmarks. On LIBERO, $π_{\texttt{RL}}$ boosts few-shot SFT models $π_0$ and $π_{0.5}$ from 57.6\% to 97.6\% and from 77.1\% to 98.3\%, respectively. On ManiSkill, we train $π_{\texttt{RL}}$ in 320 parallel environments, improving $π_0$ from 38.4\% to 78.8\% and $π_{0.5}$ from 40.1\% to 90.8\% across 4352 variations of pick-and-place task. On MetaWorld, RL is conducted over 50 different manipulation tasks and yields performance gains of 35.0\% and 26.9\% for $π_0$ and $π_{0.5}$ models, respectively. Overall, $π_{\texttt{RL}}$ achieves significant performance gains and stronger generalization over SFT-models, validating the effectiveness of online RL for flow-based VLAs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Language Model Preference Evaluation with Multiple Weak Evaluators
**Date:** 2025-10-31 | **Arxiv:** [2410.12869](https://hub.bitwiki.org/t/language-model-preference-evaluation-with-multiple-weak-evaluators/20714)

#### Abstract
Despite the remarkable success of Large Language Models (LLMs), evaluating their outputs' quality regarding preference remains a critical challenge. While existing works usually leverage a strong LLM as the judge for comparing LLMs' response pairwisely, such a single-evaluator approach is vulnerable to cyclic preference, i.e., output A is better than B, B than C, but C is better than A, causing contradictory evaluation results. To address this, we introduce PGED (Preference Graph Ensemble and Denoise), a novel approach that leverages multiple model-based evaluators to construct preference graphs, and then ensembles and denoises these graphs for acyclic, non-contradictory evaluation results. We provide theoretical guarantees for our framework, demonstrating its efficacy in recovering the ground truth preference structure. Extensive experiments on ten benchmarks demonstrate PGED 's superiority in three applications: 1) model ranking for evaluation, 2) response selection for test-time scaling, and 3) data selection for model fine-tuning. Notably, PGED combines small LLM evaluators (e.g., Llama3-8B, Mistral-7B, Qwen2-7B) to outperform strong ones (e.g., Qwen2-72B), showcasing its effectiveness in enhancing evaluation reliability and improving model performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Model Inversion with Layer-Specific Modeling and Alignment for Data-Free Continual Learning
**Date:** 2025-10-31 | **Arxiv:** [2510.26311](https://hub.bitwiki.org/t/model-inversion-with-layer-specific-modeling-and-alignment-for-data-free-continual-learning/20567)

#### Abstract
Continual learning (CL) aims to incrementally train a model on a sequence of tasks while retaining performance on prior ones. However, storing and replaying data is often infeasible due to privacy or security constraints and impractical for arbitrary pre-trained models. Data-free CL seeks to update models without access to previous data. Beyond regularization, we employ model inversion to synthesize data from the trained model, enabling replay without storing samples. Yet, model inversion in predictive models faces two challenges: (1) generating inputs solely from compressed output labels causes drift between synthetic and real data, and replaying such data can erode prior knowledge; (2) inversion is computationally expensive since each step backpropagates through the full model. These issues are amplified in large pre-trained models such as CLIP. To improve efficiency, we propose Per-layer Model Inversion (PMI), inspired by faster convergence in single-layer optimization. PMI provides strong initialization for full-model inversion, substantially reducing iterations. To mitigate feature shift, we model class-wise features via Gaussian distributions and contrastive model, ensuring alignment between synthetic and real features. Combining PMI and feature modeling, our approach enables continual learning of new classes by generating pseudo-images from semantic-aware projected features, achieving strong effectiveness and compatibility across multiple CL settings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, storing and replaying data is often infeasible due to privacy or security constraints and impractical for arbitrary pre-trained models.
* **Signal Tags:** #ai

---


### InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models
**Date:** 2025-10-23 | **Arxiv:** [2505.13878](https://hub.bitwiki.org/t/infifpo-implicit-model-fusion-via-preference-optimization-in-large-language-models/19020)

#### Abstract
Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introducing probability clipping and max-margin fusion strategies, InfiFPO enables the pivot model to align with human preferences while effectively distilling knowledge from source models. Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods. When using Phi-4 as the pivot model, InfiFPO improve its average performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its capabilities in mathematics, coding, and reasoning tasks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Alignment is Localized: A Causal Probe into Preference Layers
**Date:** 2025-10-21 | **Arxiv:** [2510.16167](https://hub.bitwiki.org/t/alignment-is-localized-a-causal-probe-into-preference-layers/18055)

#### Abstract
Reinforcement Learning frameworks, particularly those utilizing human annotations, have become an increasingly popular method for preference fine-tuning, where the outputs of a language model are tuned to match a certain set of behavioral policies or guidelines. Reinforcement Learning through Human Feedback (RLHF) is perhaps the most popular implementation of such a framework, particularly for aligning LMs toward safety and human intent. However, the internal workings of how such alignment is achieved remain largely opaque. In this work, we systematically analyze preference optimization for language model alignment by applying layer-wide causal patching between a base model and its tuned counterpart across human preference pairs. We implement our methodology on \textit{Llama-3.2-1B}, and find that alignment is spatially localized: mid-layer activations encode a distinct subspace that causally determines reward-consistent behavior, while early and late layers remain largely unaffected. Utilizing LASSO regression, we also find that only a small number of layers possess non-zero coefficients linking activation distances to reward gains. Overall, we show that, at least for some language models, alignment from human-based, preferential tuning is a directional, low rank process, rather than diffuse and parameteric.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, the internal workings of how such alignment is achieved remain largely opaque.
* **Signal Tags:** #ai

---


### SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe
**Date:** 2025-10-17 | **Arxiv:** [2410.05248](https://hub.bitwiki.org/t/sftmix-elevating-language-model-instruction-tuning-with-mixup-recipe/17641)

#### Abstract
To acquire instruction-following capabilities, large language models (LLMs) undergo instruction tuning, where they are trained on instruction-response pairs using next-token prediction (NTP). Efforts to improve instruction tuning often focus on higher-quality supervised fine-tuning (SFT) datasets, typically requiring data filtering with proprietary LLMs or human annotation. In this paper, we take a different approach by proposing SFTMix, a novel Mixup-based recipe that elevates LLM instruction tuning without relying on well-curated datasets. We observe that LLMs exhibit uneven confidence across the semantic representation space. We argue that examples with different confidence levels should play distinct roles in instruction tuning: Confident data is prone to overfitting, while unconfident data is harder to generalize. Based on this insight, SFTMix leverages training dynamics to identify examples with varying confidence levels. We then interpolate them to bridge the confidence gap and apply a Mixup-based regularization to support learning on these additional, interpolated examples. We demonstrate the effectiveness of SFTMix in both instruction-following and healthcare-specific SFT tasks, with consistent improvements across LLM families and SFT datasets of varying sizes and qualities. Extensive analyses across six directions highlight SFTMix's compatibility with data selection, adaptability to compute-constrained scenarios, and scalability to broader applications.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models
**Date:** 2025-10-15 | **Arxiv:** [2510.09976](https://hub.bitwiki.org/t/reinforcement-fine-tuning-of-flow-matching-policies-for-vision-language-action-models/16718)

#### Abstract
Vision-Language-Action (VLA) models such as OpenVLA, Octo, and $π_0$ have shown strong generalization by leveraging large-scale demonstrations, yet their performance is still fundamentally constrained by the quality and coverage of supervised data. Reinforcement learning (RL) provides a promising path for improving and fine-tuning VLAs through online interaction. However, conventional policy gradient methods are computationally infeasible in the context of flow-matching based models due to the intractability of the importance sampling process, which requires explicit computation of policy ratios. To overcome this limitation, we propose Flow Policy Optimization (FPO) algorithm, which reformulates importance sampling by leveraging per-sample changes in the conditional flow-matching objective. Furthermore, FPO achieves stable and scalable online reinforcement fine-tuning of the $π_0$ model by integrating structure-aware credit assignment to enhance gradient efficiency, clipped surrogate objectives to stabilize optimization, multi-step latent exploration to encourage diverse policy updates, and a Q-ensemble mechanism to provide robust value estimation. We evaluate FPO on the LIBERO benchmark and the ALOHA simulation task against supervised, preference-aligned, diffusion-based, autoregressive online RL, and $π_0$-FAST baselines, observing consistent improvements over the imitation prior and strong alternatives with stable learning under sparse rewards. In addition, ablation studies and analyses of the latent space dynamics further highlight the contributions of individual components within FPO, validating the effectiveness of the proposed computational modules and the stable convergence of the conditional flow-matching objective during online RL.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, conventional policy gradient methods are computationally infeasible in the context of flow-matching based models due to the intractability of the importance sampling process, which requires explicit computation of policy ratios.
* **Signal Tags:** #ai

---


### Noise Injection Systemically Degrades Large Language Model Safety Guardrails
**Date:** 2025-10-15 | **Arxiv:** [2505.13500](https://hub.bitwiki.org/t/noise-injection-systemically-degrades-large-language-model-safety-guardrails/17088)

#### Abstract
Safety guardrails in large language models (LLMs) are a critical component in preventing harmful outputs. Yet, their resilience under perturbation remains poorly understood. In this paper, we investigate the robustness of safety fine-tuning in LLMs by systematically injecting Gaussian noise into model activations. We show across multiple open-weight models that (1) Gaussian noise raises harmful-output rates (p < 0.001) by up to 27%, (2) that deeper safety fine-tuning affords no extra protection, and (3) that chain-of-thought reasoning remains largely intact. The findings reveal critical vulnerabilities in current safety alignment techniques and highlight the potential of reasoning-based and reinforcement learning approaches as promising direction for developing more robust AI safety systems. These results have important implications for real-world deployment of LLMs in safety-critical applications as these results imply that widely-deployed safety tuning methods can fail even without adversarial prompts.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models
**Date:** 2025-10-08 | **Arxiv:** [2507.12428](https://hub.bitwiki.org/t/can-we-predict-alignment-before-models-finish-thinking-towards-monitoring-misaligned-reasoning-models/15547)

#### Abstract
Reasoning language models improve performance on complex tasks by generating long chains of thought (CoTs), but this process can also increase harmful outputs in adversarial settings. In this work, we ask whether the long CoTs can be leveraged for predictive safety monitoring: do the reasoning traces provide early signals of final response alignment that could enable timely intervention? We evaluate a range of monitoring methods using either CoT text or activations, including highly capable large language models, fine-tuned classifiers, and humans. First, we find that a simple linear probe trained on CoT activations significantly outperforms all text-based baselines in predicting whether a final response is safe or unsafe, with an average absolute increase of 13 in F1 scores over the best-performing alternatives. CoT texts are often unfaithful and misleading, while model latents provide a more reliable predictive signal. Second, the probe can be applied to early CoT segments before the response is generated, showing that alignment signals appear before reasoning completes. Error analysis reveals that the performance gap between text classifiers and the linear probe largely stems from a subset of responses we call performative CoTs, where the reasoning consistently contradicts the final response as the CoT progresses. Our findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### From Noisy Traces to Stable Gradients: Bias-Variance Optimized Preference Optimization for Aligning Large Reasoning Models
**Date:** 2025-10-07 | **Arxiv:** [2510.05095](https://hub.bitwiki.org/t/from-noisy-traces-to-stable-gradients-bias-variance-optimized-preference-optimization-for-aligning-large-reasoning-models/15004)

#### Abstract
Large reasoning models (LRMs) generate intermediate reasoning traces before producing final answers, yielding strong gains on multi-step and mathematical tasks. Yet aligning LRMs with human preferences, a crucial prerequisite for model deployment, remains underexplored. The statistically correct objective for preference alignment requires marginalizing over reasoning traces, but this computation is intractable in practice. A common workaround optimizes a single sampled trajectory, which introduces substantial gradient variance from stochastic trace sampling. To address this challenge, we frame preference optimization for LRMs through the lens of the bias--variance trade-off and propose Bias--Variance Optimized Preference Optimization (BVPO), a simple, drop-in method that mixes two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. Our theory shows that BVPO strictly reduces trace-induced variance for any nontrivial mixture, provides a closed-form choice of the mixing weight that minimizes mean-squared error relative to the true marginal gradient, and under standard smoothness and step-size conditions, tightens classical convergence bounds for stochastic gradient descent. Empirically, BVPO improves alignment over the best baseline by up to 7.8 points on AlpacaEval~2 and 6.8 points on Arena-Hard. Despite being trained only on general conversational data, BVPO also boosts reasoning performance for base models by up to 4.0 points on the average of six math reasoning benchmarks. These results identify variance from trace sampling as a key bottleneck and demonstrate that directly optimizing the bias--variance trade-off yields more stable training and stronger overall performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### DNABERT-2: Fine-Tuning a Genomic Language Model for Colorectal Gene Enhancer Classification
**Date:** 2025-10-01 | **Arxiv:** [2509.25274](https://hub.bitwiki.org/t/dnabert-2-fine-tuning-a-genomic-language-model-for-colorectal-gene-enhancer-classification/13443)

#### Abstract
Gene enhancers control when and where genes switch on, yet their sequence diversity and tissue specificity make them hard to pinpoint in colorectal cancer. We take a sequence-only route and fine-tune DNABERT-2, a transformer genomic language model that uses byte-pair encoding to learn variable-length tokens from DNA. Using assays curated via the Johnston Cancer Research Centre at Queen's University Belfast, we assembled a balanced corpus of 2.34 million 1 kb enhancer sequences, applied summit-centered extraction and rigorous de-duplication including reverse-complement collapse, and split the data stratified by class. With a 4096-term vocabulary and a 232-token context chosen empirically, the DNABERT-2-117M classifier was trained with Optuna-tuned hyperparameters and evaluated on 350742 held-out sequences. The model reached PR-AUC 0.759, ROC-AUC 0.743, and best F1 0.704 at an optimized threshold (0.359), with recall 0.835 and precision 0.609. Against a CNN-based EnhancerNet trained on the same data, DNABERT-2 delivered stronger threshold-independent ranking and higher recall, although point accuracy was lower. To our knowledge, this is the first study to apply a second-generation genomic language model with BPE tokenization to enhancer classification in colorectal cancer, demonstrating the feasibility of capturing tumor-associated regulatory signals directly from DNA sequence alone. Overall, our results show that transformer-based genomic models can move beyond motif-level encodings toward holistic classification of regulatory elements, offering a novel path for cancer genomics. Next steps will focus on improving precision, exploring hybrid CNN-transformer designs, and validating across independent datasets to strengthen real-world utility.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### MOCHA: Multi-modal Objects-aware Cross-arcHitecture Alignment
**Date:** 2025-09-18 | **Arxiv:** [2509.14001](https://hub.bitwiki.org/t/mocha-multi-modal-objects-aware-cross-architecture-alignment/9918)

#### Abstract
Personalized object detection aims to adapt a general-purpose detector to recognize user-specific instances from only a few examples. Lightweight models often struggle in this setting due to their weak semantic priors, while large vision-language models (VLMs) offer strong object-level understanding but are too computationally demanding for real-time or on-device applications. We introduce MOCHA (Multi-modal Objects-aware Cross-arcHitecture Alignment), a distillation framework that transfers multimodal region-level knowledge from a frozen VLM teacher into a lightweight vision-only detector. MOCHA extracts fused visual and textual teacher's embeddings and uses them to guide student training through a dual-objective loss that enforces accurate local alignment and global relational consistency across regions. This process enables efficient transfer of semantics without the need for teacher modifications or textual input at inference. MOCHA consistently outperforms prior baselines across four personalized detection benchmarks under strict few-shot regimes, yielding a +10.1 average improvement, with minimal inference cost.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs
**Date:** 2025-09-03 | **Arxiv:** [2509.00930](https://hub.bitwiki.org/t/satquest-a-verifier-for-logical-reasoning-evaluation-and-reinforcement-fine-tuning-of-llms/7323)

#### Abstract
Recent advances in Large Language Models (LLMs) have demonstrated remarkable general reasoning capabilities. However, systematically evaluating and enhancing these reasoning capabilities is challenging due to the lack of controllable and scalable tools for fine-grained analysis. Existing benchmarks and datasets often lack the necessary variable control for multi-dimensional, systematic analysis and training, or have narrow problem types and formats. To address these limitations, we introduce SATQuest, a systematic verifier designed to evaluate and enhance logical reasoning in LLMs by generating diverse, Satisfiability-based logical reasoning problems directly from Conjunctive Normal Form (CNF) instances. SATQuest structures these problems along three orthogonal dimensions: instance scale, problem type, and question format, employing randomized, SAT-based problem generation and objective answer verification via PySAT. This design mitigates memorization issues, allows for nuanced insights into reasoning performance, and enables effective reinforcement fine-tuning. Our extensive evaluation of various LLMs using SATQuest identified significant limitations in their logical reasoning, particularly in generalizing beyond familiar mathematical formats. Furthermore, we show that reinforcement fine-tuning with SATQuest rewards substantially improves targeted task performance and generalizes to more complex instances, while highlighting remaining challenges in cross-format adaptation. Through these demonstrations, we showcase SATQuest's potential as a foundational tool and a valuable starting point for advancing LLM logical reasoning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, systematically evaluating and enhancing these reasoning capabilities is challenging due to the lack of controllable and scalable tools for fine-grained analysis.
* **Signal Tags:** #ai

---


### Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models
**Date:** 2025-08-13 | **Arxiv:** [2505.15130](https://hub.bitwiki.org/t/few-shot-adversarial-low-rank-fine-tuning-of-vision-language-models/3359)

#### Abstract
Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like (Low-Rank Adaptation) LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, to our knowledge the first method designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates training as a minimax optimization over low-rank adapters and adversarial perturbations, enabling robust adaptation with a small trainable footprint. Across eight datasets and two backbones (ViT-B/16 and ViT-B/32), AdvCLIP-LoRA achieves state-of-the-art performance in few-shot classification, adversarial base-to-new generalization, and cross-dataset transfer, delivering higher adversarial robustness than prompt tuning baselines without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical approach for robust adaptation of VLMs in resource-constrained settings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance.
* **Signal Tags:** #ai

---


### A Principled Loss Function for Direct Language Model Alignment
**Date:** 2025-08-12 | **Arxiv:** [2508.07137](https://hub.bitwiki.org/t/a-principled-loss-function-for-direct-language-model-alignment/2859)

#### Abstract
The alignment of large language models (LLMs) with human preferences is commonly achieved through Reinforcement Learning from Human Feedback (RLHF). Direct Preference Optimization (DPO) simplified this paradigm by establishing a direct mapping between the optimal policy and a reward function, eliminating the need for an explicit reward model. However, we argue that the DPO loss function is theoretically misaligned with its own derivation, as it promotes the indefinite maximization of a logits difference, which can lead to training instability and reward hacking. In this paper, we propose a novel loss function derived directly from the RLHF optimality condition. Our proposed loss targets a specific, finite value for the logits difference, which is dictated by the underlying reward, rather than its maximization. We provide a theoretical analysis, including a gradient-based comparison, to demonstrate that our method avoids the large gradients that plague DPO when the probability of dispreferred responses approaches zero. This inherent stability prevents reward hacking and leads to more effective alignment. We validate our approach by fine-tuning a Qwen2.5-7B model, showing significant win-rate improvements over a standard DPO baseline and achieving competitive performance against larger models like Llama-3.1-8B.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, we argue that the DPO loss function is theoretically misaligned with its own derivation, as it promotes the indefinite maximization of a logits difference, which can lead to training instability and reward hacking.
* **Signal Tags:** #ai

---
