# Vol 01 General Architecture Research
*Enriched by BITCOREOS | Phase 4 Batch 1*

---

### Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints
**Date:** 2025-10-10 | **Arxiv:** [2510.08549](https://arxiv.org/abs/2510.08549)

#### Abstract
We propose ERA, a new paradigm that constrains the sampling entropy above given thresholds by applying specially designed activations to the outputs of models. Our approach demonstrates broad effectiveness across different domains: 1) for large language models(LLMs), boosting the AIME 2025 score for Qwen2.5-Math-7B by 37.4%; 2) for continuous control reinforcement learning agents, improving performance by more than 30% over strong baselines such as SAC on the challenging HumanoidBench; 3) for image classification, enhancing ImageNet top-1 accuracy by 0.69% for ResNet-50. These gains are achieved with a computational overhead of less than 7%. Our work validates output activation as a powerful tool for entropy control, opening a new direction for designing simpler and more robust algorithms.

#### Research Highlights
- **Core Innovation:** We propose ERA, a new paradigm that constrains the sampling entropy above given thresholds by applying specially designed activations to the outputs of models.
- **Methodology:** See abstract.
- **Key Finding:** Our approach demonstrates broad effectiveness across different domains: 1) for large language models(LLMs), boosting the AIME 2025 score for Qwen2.5-Math-7B by 37.4%; 2) for continuous control reinforcement learning agents, improving performance by more than 30% over strong baselines such as SAC on the challenging HumanoidBench; 3) for image classification, enhancing ImageNet top-1 accuracy by 0.69% for ResNet-50.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Model
**Date:** 2025-09-29 | **Arxiv:** [2509.22518](https://arxiv.org/abs/2509.22518)

#### Abstract
Understanding how Large Language Models (LLMs) perform complex reasoning and their failure mechanisms is a challenge in interpretability research. To provide a measurable geometric analysis perspective, we define the concept of the Reasoning Manifold, a latent low-dimensional geometric structure formed by the internal representations corresponding to all correctly reasoned generations. This structure can be conceptualized as the embodiment of the effective thinking paths that the model has learned to successfully solve a given task. Based on this concept, we build REMA, a framework that explains the origins of failures by quantitatively comparing the spatial relationships of internal model representations corresponding to both erroneous and correct reasoning samples. Specifically, REMA first quantifies the geometric deviation of each erroneous representation by calculating its k-nearest neighbors distance to the approximated manifold formed by correct representations, thereby providing a unified failure signal. It then localizes the divergence points where these deviations first become significant by tracking this deviation metric across the model's layers and comparing it against a baseline of internal fluctuations from correct representations, thus identifying where the reasoning chain begins to go off-track. Our extensive experiments on diverse language and multimodal models and tasks demonstrate the low-dimensional nature of the reasoning manifold and the high separability between erroneous and correct reasoning representations. The results also validate the effectiveness of the REMA framework in analyzing the origins of reasoning failures. This research connects abstract reasoning failures to measurable geometric deviations in representations, providing new avenues for in-depth understanding and diagnosis of the internal computational processes of black-box models.

#### Research Highlights
- **Core Innovation:** Understanding how Large Language Models (LLMs) perform complex reasoning and their failure mechanisms is a challenge in interpretability research.
- **Methodology:** This research connects abstract reasoning failures to measurable geometric deviations in representations, providing new avenues for in-depth understanding and diagnosis of the internal computational processes of black-box models..
- **Key Finding:** The results also validate the effectiveness of the REMA framework in analyzing the origins of reasoning failures.

#### Technical Context
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
* **Limits:** challenge in interpretability research.
* **Signal Tags:** #ai #research

---


### An Invariant Latent Space Perspective on Language Model Inversion
**Date:** 2025-11-26 | **Arxiv:** [2511.19569](https://arxiv.org/abs/2511.19569)

#### Abstract
Language model inversion (LMI), i.e., recovering hidden prompts from outputs, emerges as a concrete threat to user privacy and system security. We recast LMI as reusing the LLM's own latent space and propose the Invariant Latent Space Hypothesis (ILSH): (1) diverse outputs from the same source prompt should preserve consistent semantics (source invariance), and (2) input<->output cyclic mappings should be self-consistent within a shared latent space (cyclic invariance). Accordingly, we present Inv^2A, which treats the LLM as an invariant decoder and learns only a lightweight inverse encoder that maps outputs to a denoised pseudo-representation. When multiple outputs are available, they are sparsely concatenated at the representation layer to increase information density. Training proceeds in two stages: contrastive alignment (source invariance) and supervised reinforcement (cyclic invariance). An optional training-free neighborhood search can refine local performance. Across 9 datasets covering user and system prompt scenarios, Inv^2A outperforms baselines by an average of 4.77% BLEU score while reducing dependence on large inverse corpora. Our analysis further shows that prevalent defenses provide limited protection, underscoring the need for stronger strategies. The source code and data involved in this paper can be found in https://github.com/yyy01/Invariant_Attacker.

#### Research Highlights
- **Core Innovation:** We recast LMI as reusing the LLM's own latent space and propose the Invariant Latent Space Hypothesis (ILSH): (1) diverse outputs from the same source prompt should preserve consistent semantics (source invariance), and (2) input<->output cyclic mappings should be self-consistent within a shared latent space (cyclic invariance).
- **Methodology:** We recast LMI as reusing the LLM's own latent space and propose the Invariant Latent Space Hypothesis (ILSH): (1) diverse outputs from the same source prompt should preserve consistent semantics (source invariance), and (2) input<->output cyclic mappings should be self-consistent within a shared latent space (cyclic invariance).
- **Key Finding:** Our analysis further shows that prevalent defenses provide limited protection, underscoring the need for stronger strategies.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Theory
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Temporal Latent Variable Structural Causal Model for Causal Discovery under External Interferences
**Date:** 2025-11-14 | **Arxiv:** [2511.10031](https://arxiv.org/abs/2511.10031)

#### Abstract
Inferring causal relationships from observed data is an important task, yet it becomes challenging when the data is subject to various external interferences. Most of these interferences are the additional effects of external factors on observed variables. Since these external factors are often unknown, we introduce latent variables to represent these unobserved factors that affect the observed data. Specifically, to capture the causal strength and adjacency information, we propose a new temporal latent variable structural causal model, incorporating causal strength and adjacency coefficients that represent the causal relationships between variables. Considering that expert knowledge can provide information about unknown interferences in certain scenarios, we develop a method that facilitates the incorporation of prior knowledge into parameter learning based on Variational Inference, to guide the model estimation. Experimental results demonstrate the stability and accuracy of our proposed method.

#### Research Highlights
- **Core Innovation:** Experimental results demonstrate the stability and accuracy of our proposed method..
- **Methodology:** See abstract.
- **Key Finding:** Experimental results demonstrate the stability and accuracy of our proposed method..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Kunlun Anomaly Troubleshooter: Enabling Kernel-Level Anomaly Detection and Causal Reasoning for Large Model Distributed Inference
**Date:** 2025-11-11 | **Arxiv:** [2511.05978](https://arxiv.org/abs/2511.05978)

#### Abstract
Anomaly troubleshooting for large model distributed inference (LMDI) remains a critical challenge. Resolving anomalies such as inference performance degradation or latency jitter in distributed system demands significant manual efforts from domain experts, resulting in extremely time-consuming diagnosis processes with relatively low accuracy. In this paper, we introduce Kunlun Anomaly Troubleshooter (KAT), the first anomaly troubleshooting framework tailored for LMDI. KAT addresses this problem through two core innovations. First, KAT exploits the synchronicity and consistency of GPU workers, innovatively leverages function trace data to precisely detect kernel-level anomalies and associated hardware components at nanosecond resolution. Second, KAT integrates these detection results into a domain-adapted LLM, delivering systematic causal reasoning and natural language interpretation of complex anomaly symptoms. Evaluations conducted in Alibaba Cloud Service production environment indicate that KAT achieves over 0.884 precision and 0.936 recall in anomaly detection, providing detail anomaly insights that significantly narrow down the diagnostic scope and improve both the efficiency and success rate of troubleshooting.

#### Research Highlights
- **Core Innovation:** In this paper, we introduce Kunlun Anomaly Troubleshooter (KAT), the first anomaly troubleshooting framework tailored for LMDI.
- **Methodology:** In this paper, we introduce Kunlun Anomaly Troubleshooter (KAT), the first anomaly troubleshooting framework tailored for LMDI.
- **Key Finding:** Second, KAT integrates these detection results into a domain-adapted LLM, delivering systematic causal reasoning and natural language interpretation of complex anomaly symptoms.

#### Technical Context
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
* **Limits:** challenge.
* **Signal Tags:** #ai #research

---


### SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning
**Date:** 2025-10-30 | **Arxiv:** [2505.16368](https://arxiv.org/abs/2505.16368)

#### Abstract
How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard.   To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLMs reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions.   We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research.

#### Research Highlights
- **Core Innovation:**   We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty.
- **Methodology:**   To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLMs reasoning.
- **Key Finding:** We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Application
* **Limits:** limitations: (1) Scalability.
* **Signal Tags:** #ai #research

---


### GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model
**Date:** 2025-10-21 | **Arxiv:** [2406.18572](https://arxiv.org/abs/2406.18572)

#### Abstract
This work tackles the problem of geo-localization with a new paradigm using a large vision-language model (LVLM) augmented with human inference knowledge. A primary challenge here is the scarcity of data for training the LVLM - existing street-view datasets often contain numerous low-quality images lacking visual clues, and lack any reasoning inference. To address the data-quality issue, we devise a CLIP-based network to quantify the degree of street-view images being locatable, leading to the creation of a new dataset comprising highly locatable street views. To enhance reasoning inference, we integrate external knowledge obtained from real geo-localization games, tapping into valuable human inference capabilities. The data are utilized to train GeoReasoner, which undergoes fine-tuning through dedicated reasoning and location-tuning stages. Qualitative and quantitative evaluations illustrate that GeoReasoner outperforms counterpart LVLMs by more than 25% at country-level and 38% at city-level geo-localization tasks, and surpasses StreetCLIP performance while requiring fewer training resources. The data and code are available at https://github.com/lingli1996/GeoReasoner.

#### Research Highlights
- **Core Innovation:** This work tackles the problem of geo-localization with a new paradigm using a large vision-language model (LVLM) augmented with human inference knowledge.
- **Methodology:** This work tackles the problem of geo-localization with a new paradigm using a large vision-language model (LVLM) augmented with human inference knowledge.
- **Key Finding:** The data and code are available at https://github.com/lingli1996/GeoReasoner..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Theory
* **Limits:** challenge here is the scarcity of data for training the LVLM - existing street-view datasets often contain numerous low-quality images lacking visual clues, and lack any reasoning inference.
* **Signal Tags:** #ai #research

---


### I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models
**Date:** 2025-10-21 | **Arxiv:** [2510.17496](https://arxiv.org/abs/2510.17496)

#### Abstract
We introduce I-RAVEN-X, a symbolic benchmark designed to evaluate generalization and robustness in analogical and mathematical reasoning for Large Language Models (LLMs) and Large Reasoning Models (LRMs). I-RAVEN-X extends I-RAVEN by increasing operand complexity, attribute range, and introducing perceptual uncertainty. Compared to LLMs, empirical results show that LRMs achieve improved productivity and systematicity on longer reasoning relations and wider attribute ranges, respectively. However, LRMs are still significantly challenged by reasoning under uncertainty and cannot effectively explore multiple probabilistic outcomes.

#### Research Highlights
- **Core Innovation:** We introduce I-RAVEN-X, a symbolic benchmark designed to evaluate generalization and robustness in analogical and mathematical reasoning for Large Language Models (LLMs) and Large Reasoning Models (LRMs).
- **Methodology:** See abstract.
- **Key Finding:** Compared to LLMs, empirical results show that LRMs achieve improved productivity and systematicity on longer reasoning relations and wider attribute ranges, respectively.

#### Technical Context
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
* **Limits:** However, LRMs are still significantly challenged by reasoning under uncertainty and cannot effectively explore multiple probabilistic outcomes.
* **Signal Tags:** #ai #research

---


### Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills
**Date:** 2025-10-15 | **Arxiv:** [2506.12963](https://arxiv.org/abs/2506.12963)

#### Abstract
Recent advances in large reasoning models (LRMs) have enabled strong chain-of-thought (CoT) generation through test-time computation. While these multi-step reasoning capabilities represent a major milestone in language model performance, they also introduce new safety risks. In this work, we present the first systematic study to revisit the problem of machine unlearning in the context of LRMs. Machine unlearning refers to the process of removing the influence of sensitive, harmful, or undesired data or knowledge from a trained model without full retraining. We show that conventional unlearning algorithms, originally designed for non-reasoning models, are inadequate for LRMs. In particular, even when final answers are successfully erased, sensitive information often persists within the intermediate reasoning steps, i.e., CoT trajectories. To address this challenge, we extend conventional unlearning and propose Reasoning-aware Representation Misdirection for Unlearning ($R^2MU$), a novel method that effectively suppresses sensitive reasoning traces and prevents the generation of associated final answers, while preserving the model's reasoning ability. Our experiments demonstrate that $R^2MU$ significantly reduces sensitive information leakage within reasoning traces and achieves strong performance across both safety and reasoning benchmarks, evaluated on state-of-the-art models such as DeepSeek-R1-Distill-LLaMA-8B and DeepSeek-R1-Distill-Qwen-14B.

#### Research Highlights
- **Core Innovation:** To address this challenge, we extend conventional unlearning and propose Reasoning-aware Representation Misdirection for Unlearning ($R^2MU$), a novel method that effectively suppresses sensitive reasoning traces and prevents the generation of associated final answers, while preserving the model's reasoning ability.
- **Methodology:** See abstract.
- **Key Finding:** Our experiments demonstrate that $R^2MU$ significantly reduces sensitive information leakage within reasoning traces and achieves strong performance across both safety and reasoning benchmarks, evaluated on state-of-the-art models such as DeepSeek-R1-Distill-LLaMA-8B and DeepSeek-R1-Distill-Qwen-14B..

#### Technical Context
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
* **Limits:** challenge, we extend conventional unlearning and propose Reasoning-aware Representation Misdirection for Unlearning ($R^2MU$), a novel method that effectively suppresses sensitive reasoning traces and prevents the generation of associated final answers, while preserving the model's reasoning ability.
* **Signal Tags:** #ai #research

---


### Cognitive Load Limits in Large Language Models: Benchmarking Multi-Hop Reasoning
**Date:** 2025-09-25 | **Arxiv:** [2509.19517](https://arxiv.org/abs/2509.19517)

#### Abstract
The scaling of Large Language Models (LLMs) has exposed a critical gap between their performance on static benchmarks and their fragility in dynamic, information-rich environments. While models excel at isolated tasks, the computational limits that govern their reasoning under cognitive load remain poorly understood. In this work, we introduce a formal theory of computational cognitive load, positing that extraneous, task-irrelevant information (Context Saturation) and interference from task-switching (Attentional Residue) are key mechanisms that degrade performance. We designed the Interleaved Cognitive Evaluation (ICE), a deconfounded benchmark to systematically manipulate these load factors on challenging multi-hop reasoning tasks. A comprehensive study (N = 10 replications per item across 200 questions) revealed significant performance variations across five instruction-tuned models. Smaller open-source architectures (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2) exhibited baseline brittleness, achieving 0% accuracy (SEM = 0.0) across all conditions, including clean controls, on this high-intrinsic-load task. In contrast, Gemini-2.0-Flash-001 showed partial resilience, achieving 85% accuracy in control conditions, with a statistically significant degradation under context saturation ($β= -0.003$ per % load, $p < 0.001$). These findings provide preliminary evidence that cognitive load is a key contributor to reasoning failures, supporting theories of hallucination-as-guessing under uncertainty. We conclude that dynamic, cognitive-aware stress testing, as exemplified by the ICE benchmark, is essential for evaluating the true resilience and safety of advanced AI systems.

#### Research Highlights
- **Core Innovation:** In this work, we introduce a formal theory of computational cognitive load, positing that extraneous, task-irrelevant information (Context Saturation) and interference from task-switching (Attentional Residue) are key mechanisms that degrade performance.
- **Methodology:** See abstract.
- **Key Finding:** In contrast, Gemini-2.0-Flash-001 showed partial resilience, achieving 85% accuracy in control conditions, with a statistically significant degradation under context saturation ($β= -0.003$ per % load, $p < 0.001$).

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Discrete Prompt Tuning via Recursive Utilization of Black-box Multimodal Large Language Model for Personalized Visual Emotion Recognition
**Date:** 2025-09-08 | **Arxiv:** [2509.04480](https://arxiv.org/abs/2509.04480)

#### Abstract
Visual Emotion Recognition (VER) is an important research topic due to its wide range of applications, including opinion mining and advertisement design. Extending this capability to recognize emotions at the individual level further broadens its potential applications. Recently, Multimodal Large Language Models (MLLMs) have attracted increasing attention and demonstrated performance comparable to that of conventional VER methods. However, MLLMs are trained on large and diverse datasets containing general opinions, which causes them to favor majority viewpoints and familiar patterns. This tendency limits their performance in a personalized VER, which is crucial for practical and real-world applications, and indicates a key area for improvement. To address this limitation, the proposed method employs discrete prompt tuning inspired by the process of humans' prompt engineering to adapt the VER task to each individual. Our method selects the best natural language representation from the generated prompts and uses it to update the prompt for the realization of accurate personalized VER.

#### Research Highlights
- **Core Innovation:** To address this limitation, the proposed method employs discrete prompt tuning inspired by the process of humans' prompt engineering to adapt the VER task to each individual.
- **Methodology:** See abstract.
- **Key Finding:** Recently, Multimodal Large Language Models (MLLMs) have attracted increasing attention and demonstrated performance comparable to that of conventional VER methods.

#### Technical Context
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
* **Limits:** However, MLLMs are trained on large and diverse datasets containing general opinions, which causes them to favor majority viewpoints and familiar patterns.
* **Signal Tags:** #ai #research

---


### Manifold Percolation: from generative model to Reinforce learning
**Date:** 2025-11-26 | **Arxiv:** [2511.20503](https://arxiv.org/abs/2511.20503)

#### Abstract
Generative modeling is typically framed as learning mapping rules, but from an observer's perspective without access to these rules, the task becomes disentangling the geometric support from the probability distribution. We propose that continuum percolation is uniquely suited to this support analysis, as the sampling process effectively projects high-dimensional density estimation onto a geometric counting problem on the support. In this work, we establish a rigorous correspondence between the topological phase transitions of random geometric graphs and the underlying data manifold in high-dimensional space. By analyzing the relationship between our proposed Percolation Shift metric and FID, we show that this metric captures structural pathologies, such as implicit mode collapse, where standard statistical metrics fail. Finally, we translate this topological phenomenon into a differentiable loss function that guides training. Experimental results confirm that this approach not only prevents manifold shrinkage but also fosters a form of synergistic improvement, where topological stability becomes a prerequisite for sustained high fidelity in both static generation and sequential decision making.

#### Research Highlights
- **Core Innovation:** By analyzing the relationship between our proposed Percolation Shift metric and FID, we show that this metric captures structural pathologies, such as implicit mode collapse, where standard statistical metrics fail.
- **Methodology:** See abstract.
- **Key Finding:** Experimental results confirm that this approach not only prevents manifold shrinkage but also fosters a form of synergistic improvement, where topological stability becomes a prerequisite for sustained high fidelity in both static generation and sequential decision making..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### TopoPerception: A Shortcut-Free Evaluation of Global Visual Perception in Large Vision-Language Models
**Date:** 2025-11-18 | **Arxiv:** [2511.11831](https://arxiv.org/abs/2511.11831)

#### Abstract
Large Vision-Language Models (LVLMs) typically align visual features from an encoder with a pre-trained Large Language Model (LLM). However, this makes the visual perception module a bottleneck, which constrains the overall capabilities of LVLMs. Conventional evaluation benchmarks, while rich in visual semantics, often contain unavoidable local shortcuts that can lead to an overestimation of models' perceptual abilities. Here, we introduce TopoPerception, a benchmark that leverages topological properties to rigorously evaluate the global visual perception capabilities of LVLMs across various granularities. Since topology depends on the global structure of an image and is invariant to local features, TopoPerception enables a shortcut-free assessment of global perception, fundamentally distinguishing it from semantically rich tasks. We evaluate state-of-the-art models on TopoPerception and find that even at the coarsest perceptual granularity, all models perform no better than random chance, indicating a profound inability to perceive global visual features. Notably, a consistent trend emerge within model families: more powerful models with stronger reasoning capabilities exhibit lower accuracy. This suggests that merely scaling up models is insufficient to address this deficit and may even exacerbate it. Progress may require new training paradigms or architectures. TopoPerception not only exposes a critical bottleneck in current LVLMs but also offers a lens and direction for improving their global visual perception. The data and code are publicly available at: https://github.com/Wenhao-Zhou/TopoPerception.

#### Research Highlights
- **Core Innovation:** Here, we introduce TopoPerception, a benchmark that leverages topological properties to rigorously evaluate the global visual perception capabilities of LVLMs across various granularities.
- **Methodology:** See abstract.
- **Key Finding:** The data and code are publicly available at: https://github.com/Wenhao-Zhou/TopoPerception..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Application
* **Limits:** However, this makes the visual perception module a bottleneck, which constrains the overall capabilities of LVLMs.
* **Signal Tags:** #ai #research

---


### SSR: Socratic Self-Refine for Large Language Model Reasoning
**Date:** 2025-11-14 | **Arxiv:** [2511.10621](https://arxiv.org/abs/2511.10621)

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable reasoning abilities, yet existing test-time frameworks often rely on coarse self-verification and self-correction, limiting their effectiveness on complex tasks. In this paper, we propose Socratic Self-Refine (SSR), a novel framework for fine-grained evaluation and precise refinement of LLM reasoning. Our proposed SSR decomposes model responses into verifiable (sub-question, sub-answer) pairs, enabling step-level confidence estimation through controlled re-solving and self-consistency checks. By pinpointing unreliable steps and iteratively refining them, SSR produces more accurate and interpretable reasoning chains. Empirical results across five reasoning benchmarks and three LLMs show that SSR consistently outperforms state-of-the-art iterative self-refinement baselines. Beyond performance gains, SSR provides a principled black-box approach for evaluating and understanding the internal reasoning processes of LLMs. Code is available at https://github.com/SalesforceAIResearch/socratic-self-refine-reasoning.

#### Research Highlights
- **Core Innovation:** Our proposed SSR decomposes model responses into verifiable (sub-question, sub-answer) pairs, enabling step-level confidence estimation through controlled re-solving and self-consistency checks.
- **Methodology:** In this paper, we propose Socratic Self-Refine (SSR), a novel framework for fine-grained evaluation and precise refinement of LLM reasoning.
- **Key Finding:** Empirical results across five reasoning benchmarks and three LLMs show that SSR consistently outperforms state-of-the-art iterative self-refinement baselines.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Application
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Planning in Branch-and-Bound: Model-Based Reinforcement Learning for Exact Combinatorial Optimization
**Date:** 2025-11-13 | **Arxiv:** [2511.09219](https://arxiv.org/abs/2511.09219)

#### Abstract
Mixed-Integer Linear Programming (MILP) lies at the core of many real-world combinatorial optimization (CO) problems, traditionally solved by branch-and-bound (B&B). A key driver influencing B&B solvers efficiency is the variable selection heuristic that guides branching decisions. Looking to move beyond static, hand-crafted heuristics, recent work has explored adapting traditional reinforcement learning (RL) algorithms to the B&B setting, aiming to learn branching strategies tailored to specific MILP distributions. In parallel, RL agents have achieved remarkable success in board games, a very specific type of combinatorial problems, by leveraging environment simulators to plan via Monte Carlo Tree Search (MCTS). Building on these developments, we introduce Plan-and-Branch-and-Bound (PlanB&B), a model-based reinforcement learning (MBRL) agent that leverages a learned internal model of the B&B dynamics to discover improved branching strategies. Computational experiments empirically validate our approach, with our MBRL branching agent outperforming previous state-of-the-art RL methods across four standard MILP benchmarks.

#### Research Highlights
- **Core Innovation:** Building on these developments, we introduce Plan-and-Branch-and-Bound (PlanB&B), a model-based reinforcement learning (MBRL) agent that leverages a learned internal model of the B&B dynamics to discover improved branching strategies.
- **Methodology:** In parallel, RL agents have achieved remarkable success in board games, a very specific type of combinatorial problems, by leveraging environment simulators to plan via Monte Carlo Tree Search (MCTS).
- **Key Finding:** Computational experiments empirically validate our approach, with our MBRL branching agent outperforming previous state-of-the-art RL methods across four standard MILP benchmarks..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### DynaAct: Large Language Model Reasoning with Dynamic Action Spaces
**Date:** 2025-11-12 | **Arxiv:** [2511.08043](https://arxiv.org/abs/2511.08043)

#### Abstract
In modern sequential decision-making systems, the construction of an optimal candidate action space is critical to efficient inference. However, existing approaches either rely on manually defined action spaces that lack scalability or utilize unstructured spaces that render exhaustive search computationally prohibitive. In this paper, we propose a novel framework named \textsc{DynaAct} for automatically constructing a compact action space to enhance sequential reasoning in complex problem-solving scenarios. Our method first estimates a proxy for the complete action space by extracting general sketches observed in a corpus covering diverse complex reasoning problems using large language models. We then formulate a submodular function that jointly evaluates candidate actions based on their utility to the current state and their diversity, and employ a greedy algorithm to select an optimal candidate set. Extensive experiments on six diverse standard benchmarks demonstrate that our approach significantly improves overall performance, while maintaining efficient inference without introducing substantial latency. The implementation is available at https://github.com/zhaoxlpku/DynaAct.

#### Research Highlights
- **Core Innovation:** In this paper, we propose a novel framework named \textsc{DynaAct} for automatically constructing a compact action space to enhance sequential reasoning in complex problem-solving scenarios.
- **Methodology:** Our method first estimates a proxy for the complete action space by extracting general sketches observed in a corpus covering diverse complex reasoning problems using large language models.
- **Key Finding:** Extensive experiments on six diverse standard benchmarks demonstrate that our approach significantly improves overall performance, while maintaining efficient inference without introducing substantial latency.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Application
* **Limits:** However, existing approaches either rely on manually defined action spaces that lack scalability or utilize unstructured spaces that render exhaustive search computationally prohibitive.
* **Signal Tags:** #ai #research

---


### Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors
**Date:** 2025-11-12 | **Arxiv:** [2505.11770](https://arxiv.org/abs/2505.11770)

#### Abstract
Interpretability research now offers a variety of techniques for identifying abstract internal mechanisms in neural networks. Can such techniques be used to predict how models will behave on out-of-distribution examples? In this work, we provide a positive answer to this question. Through a diverse set of language modeling tasks--including symbol manipulation, knowledge retrieval, and instruction following--we show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior. Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions). Both achieve high AUC-ROC in distribution and outperform methods that rely on causal-agnostic features in out-of-distribution settings, where predicting model behaviors is more crucial. Our work thus highlights a novel and significant application for internal causal analysis of language models.

#### Research Highlights
- **Core Innovation:** Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions).
- **Methodology:** Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions).
- **Key Finding:** Through a diverse set of language modeling tasks--including symbol manipulation, knowledge retrieval, and instruction following--we show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior.

#### Technical Context
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
* **Layer:** Application
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Causal Regime Detection in Energy Markets With Augmented Time Series Structural Causal Models
**Date:** 2025-11-07 | **Arxiv:** [2511.04361](https://arxiv.org/abs/2511.04361)

#### Abstract
Energy markets exhibit complex causal relationships between weather patterns, generation technologies, and price formation, with regime changes occurring continuously rather than at discrete break points. Current approaches model electricity prices without explicit causal interpretation or counterfactual reasoning capabilities. We introduce Augmented Time Series Causal Models (ATSCM) for energy markets, extending counterfactual reasoning frameworks to multivariate temporal data with learned causal structure. Our approach models energy systems through interpretable factors (weather, generation mix, demand patterns), rich grid dynamics, and observable market variables. We integrate neural causal discovery to learn time-varying causal graphs without requiring ground truth DAGs. Applied to real-world electricity price data, ATSCM enables novel counterfactual queries such as "What would prices be under different renewable generation scenarios?".

#### Research Highlights
- **Core Innovation:** We introduce Augmented Time Series Causal Models (ATSCM) for energy markets, extending counterfactual reasoning frameworks to multivariate temporal data with learned causal structure.
- **Methodology:** We introduce Augmented Time Series Causal Models (ATSCM) for energy markets, extending counterfactual reasoning frameworks to multivariate temporal data with learned causal structure.
- **Key Finding:** Applied to real-world electricity price data, ATSCM enables novel counterfactual queries such as "What would prices be under different renewable generation scenarios?"..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Humains-Junior: A 3.8B Language Model Achieving GPT-4o-Level Factual Accuracy by Directed Exoskeleton Reasoning
**Date:** 2025-10-31 | **Arxiv:** [2510.25933](https://arxiv.org/abs/2510.25933)

#### Abstract
We introduce Humans-Junior, a 3.8B model that matches GPT-4o on the FACTS Grounding public subset within a $\pm 5$ pp equivalence margin.   Results. On Q1--Q500 under identical judges, GPT-4o scores 73.5% (95% CI 69.5--77.2) and Humans-Junior 72.7% (95% CI 68.7--76.5); the paired difference is 0.8 pp (bootstrap 95% CI $-3.1$ to $+4.7$; permutation $p = 0.72$; Cohen's $d = 0.023$). TOST establishes equivalence at $\pm 5$ pp (not at $\pm 3$ pp). When purchased as managed APIs, Humans-Junior's base model (Phi-3.5-mini-instruct) is $\approx 19\times$ less expensive than GPT-4o on Microsoft AI Foundry pricing; self-hosted or edge deployments can drive incremental inference cost toward zero. Measured vs estimated pricing sources are tabulated in Appendix E.   Method. Our approach combines minimal directed "Exoskeleton Reasoning" scaffolds with behavioral fine-tuning that teaches protocol compliance (epistemic discipline) rather than domain answers. Fine-tuning alone adds little; combined, they synergize (+17.7 pp, $p < 0.001$) and reduce variance ($\approx 25\%$). In prompt-only settings on frontier models (Q1--Q100; non-comparable), directed reasoning improved GPT-4o by +11.8 pp to 85.3% and Gemini-2.5-Pro by +5.0 pp to 93.3% (baseline 88.3%, $n = 100$); see Section~5.   TL;DR. A 3.8B model achieves GPT-4o-level FACTS accuracy (equivalent within $\pm 5$ pp on Q1--Q500). Cloud pricing shows $\approx 19\times$ lower cost versus GPT-4o, and self-hosted/edge deployments can approach zero marginal cost. Pricing sources are listed in Appendix E. Frontier prompt-only gains (Q1--Q100; non-comparable) and optimized-prompt exploratory results under earlier judges are summarized in Appendix F.   Keywords: Small Language Models, Factual Grounding, Directed Reasoning, Fine-Tuning, Model Alignment, Cost-Efficient AI

#### Research Highlights
- **Core Innovation:** We introduce Humans-Junior, a 3.8B model that matches GPT-4o on the FACTS Grounding public subset within a $\pm 5$ pp equivalence margin.
- **Methodology:** See abstract.
- **Key Finding:** Frontier prompt-only gains (Q1--Q100; non-comparable) and optimized-prompt exploratory results under earlier judges are summarized in Appendix F.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Reinforcement Learning for Reasoning in Large Language Models with One Training Example
**Date:** 2025-10-27 | **Arxiv:** [2504.20571](https://arxiv.org/abs/2504.20571)

#### Abstract
We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6% (8.6% improvement beyond format correction), and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7% (7.0% non-format gain). This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which contains the aforementioned example. Furthermore, RLVR with only two examples even slightly exceeds these results (MATH500: 74.8%, average: 36.6%). Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples. In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-category generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by incorporating entropy loss with an appropriate coefficient) in 1-shot RLVR training. We also further discuss related observations about format correction, label robustness and prompt modification. These findings can inspire future work on RLVR efficiency and encourage a re-examination of recent progress and the underlying mechanisms in RLVR. All resources are open source at https://github.com/ypwang61/One-Shot-RLVR.

#### Research Highlights
- **Core Innovation:** We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs).
- **Methodology:** This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which contains the aforementioned example.
- **Key Finding:** We also show the critical role of promoting exploration (e.g., by incorporating entropy loss with an appropriate coefficient) in 1-shot RLVR training.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Application
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### A recursive Bayesian neural network for constitutive modeling of sands under monotonic and cyclic loading
**Date:** 2025-10-23 | **Arxiv:** [2501.10088](https://arxiv.org/abs/2501.10088)

#### Abstract
In geotechnical engineering, constitutive models are central to capturing soil behavior across diverse drainage conditions, stress paths,and loading histories. While data driven deep learning (DL) approaches have shown promise as alternatives to traditional constitutive formulations, their deployment requires models that are both accurate and capable of quantifying predictive uncertainty. This study introduces a recursive Bayesian neural network (rBNN) framework that unifies temporal sequence learning with generalized Bayesian inference to achieve both predictive accuracy and rigorous uncertainty quantification. A key innovation is the incorporation of a sliding window recursive structure that enables the model to effectively capture path dependent soil responses under monotonic and cyclic loading. By treating network parameters as random variables and inferring their posterior distributions via generalized variational inference, the rBNN produces well calibrated confidence intervals alongside point predictions.The framework is validated against four datasets spanning both simulated and experimental triaxial tests: monotonic loading using a Hardening Soil model simulation and 28 CD tests on Baskarp sand, and cyclic loading using an exponential constitutive simulation of CD CU tests and 37 experimental cyclic CU tests on Ottawa F65 sand. This progression from monotonic to cyclic and from simulated to experimental data demonstrates the adaptability of the proposed approach across varying levels of data fidelity and complexity. Comparative analyses with LSTM, Encoder Decoder,and GRU architectures highlight that rBNN not only achieves competitive predictive accuracy but also provides reliable confidence intervals.

#### Research Highlights
- **Core Innovation:** This progression from monotonic to cyclic and from simulated to experimental data demonstrates the adaptability of the proposed approach across varying levels of data fidelity and complexity.
- **Methodology:** By treating network parameters as random variables and inferring their posterior distributions via generalized variational inference, the rBNN produces well calibrated confidence intervals alongside point predictions.The framework is validated against four datasets spanning both simulated and experimental triaxial tests: monotonic loading using a Hardening Soil model simulation and 28 CD tests on Baskarp sand, and cyclic loading using an exponential constitutive simulation of CD CU tests and 37 experimental cyclic CU tests on Ottawa F65 sand.
- **Key Finding:** This progression from monotonic to cyclic and from simulated to experimental data demonstrates the adaptability of the proposed approach across varying levels of data fidelity and complexity.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Neural Architecture
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### UniRL-Zero: Reinforcement Learning on Unified Models with Joint Language Model and Diffusion Model Experts
**Date:** 2025-10-22 | **Arxiv:** [2510.17937](https://arxiv.org/abs/2510.17937)

#### Abstract
We present UniRL-Zero, a unified reinforcement learning (RL) framework that boosts, multimodal language model understanding and reasoning, diffusion model multimedia generation, and their beneficial interaction capabilities within a unified model. Our work defines six scenarios for unified model reinforcement learning, providing systematic baselines for reinforcement learning of unified understanding and generation model. Our code is available at https://github.com/G-U-N/UniRL.

#### Research Highlights
- **Core Innovation:** We present UniRL-Zero, a unified reinforcement learning (RL) framework that boosts, multimodal language model understanding and reasoning, diffusion model multimedia generation, and their beneficial interaction capabilities within a unified model.
- **Methodology:** We present UniRL-Zero, a unified reinforcement learning (RL) framework that boosts, multimodal language model understanding and reasoning, diffusion model multimedia generation, and their beneficial interaction capabilities within a unified model.
- **Key Finding:** Our code is available at https://github.com/G-U-N/UniRL..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### CALM: A Causal Analysis Language Model for Tabular Data in Complex Systems with Local Scores, Conditional Independence Tests, and Relation Attributes
**Date:** 2025-10-15 | **Arxiv:** [2510.09846](https://arxiv.org/abs/2510.09846)

#### Abstract
Causal discovery from observational data is fundamental to scientific fields like biology, where controlled experiments are often impractical. However, existing methods, including constraint-based (e.g., PC, causalMGM) and score-based approaches (e.g., NOTEARS), face significant limitations. These include an inability to resolve causal direction, restrictions to linear associations, sensitivity to violations of the faithfulness assumption, and inefficiency in searching vast hypothesis spaces. While large language models (LLMs) offer powerful reasoning capabilities, their application is hindered by a fundamental discrepancy: they are designed for text, while most causal data is tabular. To address these challenges, we introduce CALM, a novel causal analysis language model specifically designed for tabular data in complex systems. CALM leverages a Mamba-based architecture to classify causal patterns from pairwise variable relationships. It integrates a comprehensive suite of evidence, including local causal scores, conditional independence tests, and relational attributes, to capture a wide spectrum of linear, nonlinear, and conditional causal mechanisms. Trained on a diverse corpus of synthetic data (from linear, mixed, and nonlinear models) and 10 real-world biological datasets with rigorously validated causal relationships, our model ensures robustness and generalizability. Empirical evaluation demonstrates that CALM significantly outperforms existing methods in both simulation studies, achieving over 91% accuracy, and in a real-world application identifying causal factors in Hepatitis C virus progression. This work represents a significant step towards accurate and generalizable causal discovery by successfully adapting the pattern recognition capabilities of language models to the intricacies of tabular data.

#### Research Highlights
- **Core Innovation:** To address these challenges, we introduce CALM, a novel causal analysis language model specifically designed for tabular data in complex systems.
- **Methodology:** See abstract.
- **Key Finding:** Empirical evaluation demonstrates that CALM significantly outperforms existing methods in both simulation studies, achieving over 91% accuracy, and in a real-world application identifying causal factors in Hepatitis C virus progression.

#### Technical Context
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
* **Limits:** However, existing methods, including constraint-based (e.
* **Signal Tags:** #ai #research

---


### Spatiotemporal Forecasting as Planning: A Model-Based Reinforcement Learning Approach with Generative World Models
**Date:** 2025-10-07 | **Arxiv:** [2510.04020](https://arxiv.org/abs/2510.04020)

#### Abstract
To address the dual challenges of inherent stochasticity and non-differentiable metrics in physical spatiotemporal forecasting, we propose Spatiotemporal Forecasting as Planning (SFP), a new paradigm grounded in Model-Based Reinforcement Learning. SFP constructs a novel Generative World Model to simulate diverse, high-fidelity future states, enabling an "imagination-based" environmental simulation. Within this framework, a base forecasting model acts as an agent, guided by a beam search-based planning algorithm that leverages non-differentiable domain metrics as reward signals to explore high-return future sequences. These identified high-reward candidates then serve as pseudo-labels to continuously optimize the agent's policy through iterative self-training, significantly reducing prediction error and demonstrating exceptional performance on critical domain metrics like capturing extreme events.

#### Research Highlights
- **Core Innovation:** To address the dual challenges of inherent stochasticity and non-differentiable metrics in physical spatiotemporal forecasting, we propose Spatiotemporal Forecasting as Planning (SFP), a new paradigm grounded in Model-Based Reinforcement Learning.
- **Methodology:** Within this framework, a base forecasting model acts as an agent, guided by a beam search-based planning algorithm that leverages non-differentiable domain metrics as reward signals to explore high-return future sequences.
- **Key Finding:** These identified high-reward candidates then serve as pseudo-labels to continuously optimize the agent's policy through iterative self-training, significantly reducing prediction error and demonstrating exceptional performance on critical domain metrics like capturing extreme events..

#### Technical Context
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
* **Limits:** challenges of inherent stochasticity and non-differentiable metrics in physical spatiotemporal forecasting, we propose Spatiotemporal Forecasting as Planning (SFP), a new paradigm grounded in Model-Based Reinforcement Learning.
* **Signal Tags:** #ai #research

---


### L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning
**Date:** 2025-10-06 | **Arxiv:** [2503.04697](https://arxiv.org/abs/2503.04697)

#### Abstract
Reasoning language models have shown an uncanny ability to improve performance at test-time by ``thinking longer''-that is, by generating longer chain-of-thought sequences and hence using more compute. However, the length of their chain-of-thought reasoning is not controllable, making it impossible to allocate test-time compute to achieve a desired level of performance. We introduce Length Controlled Policy Optimization (LCPO), a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints. We use LCPO to train L1, a reasoning language model that produces outputs satisfying a length constraint given in its prompt. L1's length control allows for smoothly trading off computational cost and accuracy on a wide range of tasks, and outperforms the state-of-the-art S1 method for length control. Furthermore, we uncover an unexpected short chain-of-thought capability in models trained with LCPO. Specifically, using LCPO we derive Short Reasoning Models (SRMs), that exhibit similar reasoning patterns as full-length reasoning models, but can generate CoT lengths comparable to non-reasoning models. They demonstrate significant performance gains, for instance, our 1.5B L1 model surpasses GPT-4o at equal reasoning lengths. Overall, LCPO enables precise control over reasoning length, allowing for fine-grained allocation of test-time compute and accuracy. We release code and models at https://www.cmu-l3.github.io/l1

#### Research Highlights
- **Core Innovation:** We introduce Length Controlled Policy Optimization (LCPO), a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints.
- **Methodology:** Specifically, using LCPO we derive Short Reasoning Models (SRMs), that exhibit similar reasoning patterns as full-length reasoning models, but can generate CoT lengths comparable to non-reasoning models.
- **Key Finding:** They demonstrate significant performance gains, for instance, our 1.5B L1 model surpasses GPT-4o at equal reasoning lengths.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Theory
* **Limits:** However, the length of their chain-of-thought reasoning is not controllable, making it impossible to allocate test-time compute to achieve a desired level of performance.
* **Signal Tags:** #ai #research

---


### Language Model Planning from an Information Theoretic Perspective
**Date:** 2025-10-01 | **Arxiv:** [2509.25260](https://arxiv.org/abs/2509.25260)

#### Abstract
The extent to which decoder-only language models (LMs) engage in planning, that is, organizing intermediate computations to support coherent long-range generation, remains an open and important question, with implications for interpretability, reliability, and principled model design. Planning involves structuring computations over long horizons, considering multiple possible continuations, and selectively reusing past information, but how effectively transformer-based LMs realize these capabilities is still unclear. We address these questions by analyzing the hidden states at the core of transformer computations, which capture intermediate results and act as carriers of information. Since these hidden representations are often redundant and encumbered with fine-grained details, we develop a pipeline based on vector-quantized variational autoencoders that compresses them into compact summary codes. These codes enable measuring mutual information, allowing systematic analysis of the computational structure underlying model behavior. Using this framework, we study planning in LMs across synthetic grammar, path-finding tasks, and natural language datasets, focusing on three key aspects: (i) the planning horizon of pre-output computations, (ii) the extent to which the model considers alternative valid continuations, and (iii) the reliance of new predictions on earlier computations. By answering these questions, we advance the understanding of how planning is realized in LMs and contribute a general-purpose pipeline for probing the internal dynamics of LMs and deep learning systems. Our results reveal that the effective planning horizon is task-dependent, that models implicitly preserve information about unused correct continuations, and that predictions draw most on recent computations, though earlier blocks remain informative.

#### Research Highlights
- **Core Innovation:** The extent to which decoder-only language models (LMs) engage in planning, that is, organizing intermediate computations to support coherent long-range generation, remains an open and important question, with implications for interpretability, reliability, and principled model design.
- **Methodology:** Using this framework, we study planning in LMs across synthetic grammar, path-finding tasks, and natural language datasets, focusing on three key aspects: (i) the planning horizon of pre-output computations, (ii) the extent to which the model considers alternative valid continuations, and (iii) the reliance of new predictions on earlier computations.
- **Key Finding:** Our results reveal that the effective planning horizon is task-dependent, that models implicitly preserve information about unused correct continuations, and that predictions draw most on recent computations, though earlier blocks remain informative..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Infrastructure
* **Limits:** remains an open and important question, with implications for interpretability, reliability, and principled model design.
* **Signal Tags:** #ai #research

---


### An Empirical Study of Federated Prompt Learning for Vision Language Model
**Date:** 2025-09-19 | **Arxiv:** [2505.23024](https://arxiv.org/abs/2505.23024)

#### Abstract
The Vision Language Model (VLM) excels in aligning vision and language representations, and prompt learning has emerged as a key technique for adapting such models to downstream tasks. However, the application of prompt learning with VLM in federated learning (FL) scenarios remains underexplored. This paper systematically investigates the behavioral differences between language prompt learning (LPT) and vision prompt learning (VPT) under data heterogeneity challenges, including label skew and domain shift. We conduct extensive experiments to evaluate the impact of various FL and prompt configurations, such as client scale, aggregation strategies, and prompt length, to assess the robustness of Federated Prompt Learning (FPL). Furthermore, we explore strategies for enhancing prompt learning in complex scenarios where label skew and domain shift coexist, including leveraging both prompt types when computational resources allow. Our findings offer practical insights into optimizing prompt learning in federated settings, contributing to the broader deployment of VLMs in privacy-preserving environments.

#### Research Highlights
- **Core Innovation:** The Vision Language Model (VLM) excels in aligning vision and language representations, and prompt learning has emerged as a key technique for adapting such models to downstream tasks.
- **Methodology:** See abstract.
- **Key Finding:** Our findings offer practical insights into optimizing prompt learning in federated settings, contributing to the broader deployment of VLMs in privacy-preserving environments..

#### Technical Context
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
* **Limits:** However, the application of prompt learning with VLM in federated learning (FL) scenarios remains underexplored.
* **Signal Tags:** #ai #research

---


### Learning Counterfactually Fair Models via Improved Generation with Neural Causal Models
**Date:** 2025-09-08 | **Arxiv:** [2502.12796](https://arxiv.org/abs/2502.12796)

#### Abstract
One of the main concerns while deploying machine learning models in real-world applications is fairness. Counterfactual fairness has emerged as an intuitive and natural definition of fairness. However, existing methodologies for enforcing counterfactual fairness seem to have two limitations: (i) generating counterfactual samples faithful to the underlying causal graph, and (ii) as we argue in this paper, existing regularizers are mere proxies and do not directly enforce the exact definition of counterfactual fairness. In this work, our aim is to mitigate both issues. Firstly, we propose employing Neural Causal Models (NCMs) for generating the counterfactual samples. For implementing the abduction step in NCMs, the posteriors of the exogenous variables need to be estimated given a counterfactual query, as they are not readily available. As a consequence, $\mathcal{L}_3$ consistency with respect to the underlying causal graph cannot be guaranteed in practice due to the estimation errors involved. To mitigate this issue, we propose a novel kernel least squares loss term that enforces the $\mathcal{L}_3$ constraints explicitly. Thus, we obtain an improved counterfactual generation suitable for the counterfactual fairness task. Secondly, we propose a new MMD-based regularizer term that explicitly enforces the counterfactual fairness conditions into the base model while training. We show an improved trade-off between counterfactual fairness and generalization over existing baselines on synthetic and benchmark datasets.

#### Research Highlights
- **Core Innovation:** Secondly, we propose a new MMD-based regularizer term that explicitly enforces the counterfactual fairness conditions into the base model while training.
- **Methodology:** See abstract.
- **Key Finding:** We show an improved trade-off between counterfactual fairness and generalization over existing baselines on synthetic and benchmark datasets..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Application
* **Limits:** However, existing methodologies for enforcing counterfactual fairness seem to have two limitations: (i) generating counterfactual samples faithful to the underlying causal graph, and (ii) as we argue in this paper, existing regularizers are mere proxies and do not directly enforce the exact definition of counterfactual fairness.
* **Signal Tags:** #ai #research

---


### TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning
**Date:** 2025-09-08 | **Arxiv:** [2505.11737](https://arxiv.org/abs/2505.11737)

#### Abstract
While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a Token-level Uncertainty estimation framework for Reasoning (TokUR) that enables LLMs to self-assess and self-improve their responses in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, and we aggregate these uncertainty quantities to capture the semantic uncertainty of generated responses. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that TokUR exhibits a strong correlation with answer correctness and model robustness, and the uncertainty signals produced by TokUR can be leveraged to enhance the model's reasoning performance at test time. These results highlight the effectiveness of TokUR as a principled and scalable approach for improving the reliability and interpretability of LLMs in challenging reasoning tasks.

#### Research Highlights
- **Core Innovation:** Specifically, we introduce low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, and we aggregate these uncertainty quantities to capture the semantic uncertainty of generated responses.
- **Methodology:** In this paper, we propose a Token-level Uncertainty estimation framework for Reasoning (TokUR) that enables LLMs to self-assess and self-improve their responses in mathematical reasoning.
- **Key Finding:** These results highlight the effectiveness of TokUR as a principled and scalable approach for improving the reliability and interpretability of LLMs in challenging reasoning tasks..

#### Technical Context
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
* **Limits:** remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning.
* **Signal Tags:** #ai #research

---


### Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning
**Date:** 2025-09-03 | **Arxiv:** [2502.03275](https://arxiv.org/abs/2502.03275)

#### Abstract
Large Language Models (LLMs) excel at reasoning and planning when trained on chainof-thought (CoT) data, where the step-by-step thought process is explicitly outlined by text tokens. However, this results in lengthy inputs where many words support textual coherence rather than core reasoning information, and processing these inputs consumes substantial computation resources. In this work, we propose a hybrid representation of the reasoning process, where we partially abstract away the initial reasoning steps using latent discrete tokens generated by VQ-VAE, significantly reducing the length of reasoning traces. We explore the use of latent trace abstractions in two scenarios: 1) training the model from scratch for the Keys-Finding Maze problem, 2) fine-tuning LLMs on this hybrid data with an extended vocabulary including unseen latent tokens, for both logical and mathematical reasoning problems. To facilitate effective learning, we introduce a simple training procedure that randomly mixes latent and text tokens, which enables fast adaptation to new latent tokens. Our approach consistently outperforms the baselines methods in various benchmarks.

#### Research Highlights
- **Core Innovation:** To facilitate effective learning, we introduce a simple training procedure that randomly mixes latent and text tokens, which enables fast adaptation to new latent tokens.
- **Methodology:** In this work, we propose a hybrid representation of the reasoning process, where we partially abstract away the initial reasoning steps using latent discrete tokens generated by VQ-VAE, significantly reducing the length of reasoning traces.
- **Key Finding:** However, this results in lengthy inputs where many words support textual coherence rather than core reasoning information, and processing these inputs consumes substantial computation resources.

#### Technical Context
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
* **Limits:** However, this results in lengthy inputs where many words support textual coherence rather than core reasoning information, and processing these inputs consumes substantial computation resources.
* **Signal Tags:** #ai #research

---


### NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models
**Date:** 2025-12-09 | **Arxiv:** [2512.07218](https://arxiv.org/abs/2512.07218)

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, temporal reasoning, particularly under complex temporal constraints, remains a major challenge. To this end, existing approaches have explored symbolic methods, which encode temporal structure explicitly, and reflective mechanisms, which revise reasoning errors through multi-step inference. Nonetheless, symbolic approaches often underutilize the reasoning capabilities of LLMs, while reflective methods typically lack structured temporal representations, which can result in inconsistent or hallucinated reasoning. As a result, even when the correct temporal context is available, LLMs may still misinterpret or misapply time-related information, leading to incomplete or inaccurate answers. To address these limitations, in this work, we propose Neuro-Symbolic Temporal Reasoning (NeSTR), a novel framework that integrates structured symbolic representations with hybrid reflective reasoning to enhance the temporal sensitivity of LLM inference. NeSTR preserves explicit temporal relations through symbolic encoding, enforces logical consistency via verification, and corrects flawed inferences using abductive reflection. Extensive experiments on diverse temporal question answering benchmarks demonstrate that NeSTR achieves superior zero-shot performance and consistently improves temporal reasoning without any fine-tuning, showcasing the advantage of neuro-symbolic integration in enhancing temporal understanding in large language models.

#### Research Highlights
- **Core Innovation:** To address these limitations, in this work, we propose Neuro-Symbolic Temporal Reasoning (NeSTR), a novel framework that integrates structured symbolic representations with hybrid reflective reasoning to enhance the temporal sensitivity of LLM inference.
- **Methodology:** NeSTR preserves explicit temporal relations through symbolic encoding, enforces logical consistency via verification, and corrects flawed inferences using abductive reflection.
- **Key Finding:** Extensive experiments on diverse temporal question answering benchmarks demonstrate that NeSTR achieves superior zero-shot performance and consistently improves temporal reasoning without any fine-tuning, showcasing the advantage of neuro-symbolic integration in enhancing temporal understanding in large language models..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Application
* **Limits:** However, temporal reasoning, particularly under complex temporal constraints, remains a major challenge.
* **Signal Tags:** #ai #research

---


### Model Gateway: Model Management Platform for Model-Driven Drug Discovery
**Date:** 2025-12-08 | **Arxiv:** [2512.05462](https://arxiv.org/abs/2512.05462)

#### Abstract
This paper presents the Model Gateway, a management platform for managing machine learning (ML) and scientific computational models in the drug discovery pipeline. The platform supports Large Language Model (LLM) Agents and Generative AI-based tools to perform ML model management tasks in our Machine Learning operations (MLOps) pipelines, such as the dynamic consensus model, a model that aggregates several scientific computational models, registration and management, retrieving model information, asynchronous submission/execution of models, and receiving results once the model complete executions. The platform includes a Model Owner Control Panel, Platform Admin Tools, and Model Gateway API service for interacting with the platform and tracking model execution. The platform achieves a 0% failure rate when testing scaling beyond 10k simultaneous application clients consume models. The Model Gateway is a fundamental part of our model-driven drug discovery pipeline. It has the potential to significantly accelerate the development of new drugs with the maturity of our MLOps infrastructure and the integration of LLM Agents and Generative AI tools.

#### Research Highlights
- **Core Innovation:** This paper presents the Model Gateway, a management platform for managing machine learning (ML) and scientific computational models in the drug discovery pipeline.
- **Methodology:** See abstract.
- **Key Finding:** The platform supports Large Language Model (LLM) Agents and Generative AI-based tools to perform ML model management tasks in our Machine Learning operations (MLOps) pipelines, such as the dynamic consensus model, a model that aggregates several scientific computational models, registration and management, retrieving model information, asynchronous submission/execution of models, and receiving results once the model complete executions.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Reasoning in Diffusion Large Language Models is Concentrated in Dynamic Confusion Zones
**Date:** 2025-11-20 | **Arxiv:** [2511.15208](https://arxiv.org/abs/2511.15208)

#### Abstract
Diffusion Large Language Models (dLLMs) are rapidly emerging alongside autoregressive models as a powerful paradigm for complex reasoning, with reinforcement learning increasingly used for downstream alignment. Existing trajectory-based RL methods uniformly allocate policy gradients across denoising steps, implicitly treating all steps as equally important. We challenge this assumption by analyzing trajectories with several step-level metrics: entropy-based uncertainty, Confidence-Margin (CM) uncertainty, and Rate of Entropy Change (RoEC). These reveal structured "zones of confusion": transient spikes in uncertainty and instability that strongly predict final success or failure, while most steps remain stable. We propose Adaptive Trajectory Policy Optimization (ATPO), a lightweight step-selection strategy that dynamically reallocates gradient updates to these high-leverage steps without changing the RL objective, rewards, or compute budget. Using a hybrid RoEC+CM rule, ATPO delivers substantial gains in reasoning accuracy and training stability across benchmarks, showing that exploiting trajectory dynamics is key to advancing dLLM RL.

#### Research Highlights
- **Core Innovation:** We propose Adaptive Trajectory Policy Optimization (ATPO), a lightweight step-selection strategy that dynamically reallocates gradient updates to these high-leverage steps without changing the RL objective, rewards, or compute budget.
- **Methodology:** Using a hybrid RoEC+CM rule, ATPO delivers substantial gains in reasoning accuracy and training stability across benchmarks, showing that exploiting trajectory dynamics is key to advancing dLLM RL..
- **Key Finding:** Using a hybrid RoEC+CM rule, ATPO delivers substantial gains in reasoning accuracy and training stability across benchmarks, showing that exploiting trajectory dynamics is key to advancing dLLM RL..

#### Technical Context
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
* **Limits:** challenge this assumption by analyzing trajectories with several step-level metrics: entropy-based uncertainty, Confidence-Margin (CM) uncertainty, and Rate of Entropy Change (RoEC).
* **Signal Tags:** #ai #research

---


### TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training
**Date:** 2025-11-14 | **Arxiv:** [2511.09741](https://arxiv.org/abs/2511.09741)

#### Abstract
Training large language models (LLMs) is fundamentally constrained by limited device memory and costly inter-device communication. Although pipeline parallelism alleviates memory pressure by partitioning models across devices, it incurs activation communication overhead that scales linearly with sequence length, limiting efficiency in long-context training. Recent weight-passing approaches (e.g., WeiPipe) mitigate this by transmitting model weights instead of activations, but suffer from redundant peer-to-peer (P2P) transfers and underutilized intra-node bandwidth. We propose TawPipe--topology-aware weight pipeline parallelism, which exploits hierarchical bandwidth in distributed clusters for improved communication efficiency. TawPipe: (i) groups devices based on topology to optimize intra-node collective and inter-node P2P communication; (ii) assigns each device a fixed shard of model weights and gradients, avoiding redundant transfers; and (iii) overlaps communication with computation to hide latency. Unlike global collective operations used in fully sharded data parallelism (FSDP), TawPipe confines most communication within node boundaries, significantly reducing cross-node traffic. Extensive experiments on up to 24 GPUs with LLaMA-style models show that TawPipe achieves superior throughput and scalability compared to state-of-the-art baselines.

#### Research Highlights
- **Core Innovation:** We propose TawPipe--topology-aware weight pipeline parallelism, which exploits hierarchical bandwidth in distributed clusters for improved communication efficiency.
- **Methodology:** Although pipeline parallelism alleviates memory pressure by partitioning models across devices, it incurs activation communication overhead that scales linearly with sequence length, limiting efficiency in long-context training.
- **Key Finding:** Extensive experiments on up to 24 GPUs with LLaMA-style models show that TawPipe achieves superior throughput and scalability compared to state-of-the-art baselines..

#### Technical Context
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
* **Limits:** Although pipeline parallelism alleviates memory pressure by partitioning models across devices, it incurs activation communication overhead that scales linearly with sequence length, limiting efficiency in long-context training.
* **Signal Tags:** #ai #research

---


### The Probably Approximately Correct Learning Model in Computational Learning Theory
**Date:** 2025-11-13 | **Arxiv:** [2511.08791](https://arxiv.org/abs/2511.08791)

#### Abstract
This survey paper gives an overview of various known results on learning classes of Boolean functions in Valiant's Probably Approximately Correct (PAC) learning model and its commonly studied variants.

#### Research Highlights
- **Core Innovation:** This survey paper gives an overview of various known results on learning classes of Boolean functions in Valiant's Probably Approximately Correct (PAC) learning model and its commonly studied variants..
- **Methodology:** See abstract.
- **Key Finding:** This survey paper gives an overview of various known results on learning classes of Boolean functions in Valiant's Probably Approximately Correct (PAC) learning model and its commonly studied variants..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Extrapolation to infinite model space of no-core shell model calculations using machine learning
**Date:** 2025-11-13 | **Arxiv:** [2511.05061](https://arxiv.org/abs/2511.05061)

#### Abstract
An ensemble of neural networks is employed to extrapolate no-core shell model (NCSM) results to infinite model space for light nuclei. We present a review of our neural network extrapolations of the NCSM results obtained with the Daejeon16 NN interaction in different model spaces and with different values of the NCSM basis parameter $\hbarΩ$ for energies of nuclear states and root-mean-square (rms) radii of proton, neutron and matter distributions in light nuclei. The method yields convergent predictions with quantifiable uncertainties. Ground-state energies for $^{6}$Li, $^{6}$He, and the unbound $^{6}$Be, as well as the excited $(3^{+},0)$ and $(0^{+},1)$ states of $^{6}$Li, are obtained within a few hundred keV of experiment. The extrapolated radii of bound states converge well. In contrast, radii of unbound states in $^{6}$Be and $^{6}$Li do not stabilize.

#### Research Highlights
- **Core Innovation:** An ensemble of neural networks is employed to extrapolate no-core shell model (NCSM) results to infinite model space for light nuclei.
- **Methodology:** See abstract.
- **Key Finding:** We present a review of our neural network extrapolations of the NCSM results obtained with the Daejeon16 NN interaction in different model spaces and with different values of the NCSM basis parameter $\hbarΩ$ for energies of nuclear states and root-mean-square (rms) radii of proton, neutron and matter distributions in light nuclei.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Anatomy-VLM: A Fine-grained Vision-Language Model for Medical Interpretation
**Date:** 2025-11-12 | **Arxiv:** [2511.08402](https://arxiv.org/abs/2511.08402)

#### Abstract
Accurate disease interpretation from radiology remains challenging due to imaging heterogeneity. Achieving expert-level diagnostic decisions requires integration of subtle image features with clinical knowledge. Yet major vision-language models (VLMs) treat images as holistic entities and overlook fine-grained image details that are vital for disease diagnosis. Clinicians analyze images by utilizing their prior medical knowledge and identify anatomical structures as important region of interests (ROIs). Inspired from this human-centric workflow, we introduce Anatomy-VLM, a fine-grained, vision-language model that incorporates multi-scale information. First, we design a model encoder to localize key anatomical features from entire medical images. Second, these regions are enriched with structured knowledge for contextually-aware interpretation. Finally, the model encoder aligns multi-scale medical information to generate clinically-interpretable disease prediction. Anatomy-VLM achieves outstanding performance on both in- and out-of-distribution datasets. We also validate the performance of Anatomy-VLM on downstream image segmentation tasks, suggesting that its fine-grained alignment captures anatomical and pathology-related knowledge. Furthermore, the Anatomy-VLM's encoder facilitates zero-shot anatomy-wise interpretation, providing its strong expert-level clinical interpretation capabilities.

#### Research Highlights
- **Core Innovation:** Inspired from this human-centric workflow, we introduce Anatomy-VLM, a fine-grained, vision-language model that incorporates multi-scale information.
- **Methodology:** See abstract.
- **Key Finding:** Furthermore, the Anatomy-VLM's encoder facilitates zero-shot anatomy-wise interpretation, providing its strong expert-level clinical interpretation capabilities..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Theory
* **Limits:** remains challenging due to imaging heterogeneity.
* **Signal Tags:** #ai #research

---


### Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models
**Date:** 2025-11-12 | **Arxiv:** [2511.08577](https://arxiv.org/abs/2511.08577)

#### Abstract
Improving reasoning capabilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Prior work proposes recurrent transformers, which allocate a fixed number of extra iterations per token to improve generation quality. After the first, standard forward pass, instead of verbalization, last-layer hidden states are fed back as inputs for additional iterations to refine token predictions. Yet we identify a latent overthinking phenomenon: easy token predictions that are already correct after the first pass are sometimes revised into errors in additional iterations. To address this, we propose Think-at-Hard (TaH), a dynamic latent thinking method that iterates deeper only at hard tokens. It employs a lightweight neural decider to trigger latent iterations only at tokens that are likely incorrect after the standard forward pass. During latent iterations, Low-Rank Adaptation (LoRA) modules shift the LLM objective from general next-token prediction to focused hard-token refinement. We further introduce a duo-causal attention mechanism that extends attention from the token sequence dimension to an additional iteration depth dimension. This enables cross-iteration information flow while maintaining full sequential parallelism. Experiments show that TaH boosts LLM reasoning performance across five challenging benchmarks while maintaining the same parameter count. Compared with baselines that iterate twice for all output tokens, TaH delivers 8.1-11.3% accuracy gains while exempting 94% of tokens from the second iteration. Against strong single-iteration Qwen3 models finetuned with the same data, it also delivers 4.0-5.0% accuracy gains. When allowing less than 3% additional parameters from LoRA and the iteration decider, the gains increase to 8.5-12.6% and 5.3-5.4%, respectively. Our code is available at https://github.com/thu-nics/TaH.

#### Research Highlights
- **Core Innovation:** We further introduce a duo-causal attention mechanism that extends attention from the token sequence dimension to an additional iteration depth dimension.
- **Methodology:** See abstract.
- **Key Finding:** Experiments show that TaH boosts LLM reasoning performance across five challenging benchmarks while maintaining the same parameter count.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Formal Reasoning
* **Layer:** Application
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning
**Date:** 2025-11-12 | **Arxiv:** [2505.13709](https://arxiv.org/abs/2505.13709)

#### Abstract
Offline reinforcement learning (RL) offers a powerful paradigm for data-driven control. Compared to model-free approaches, offline model-based RL (MBRL) explicitly learns a world model from a static dataset and uses it as a surrogate simulator, improving data efficiency and enabling potential generalization beyond the dataset support. However, most existing offline MBRL methods follow a two-stage training procedure: first learning a world model by maximizing the likelihood of the observed transitions, then optimizing a policy to maximize its expected return under the learned model. This objective mismatch results in a world model that is not necessarily optimized for effective policy learning. Moreover, we observe that policies learned via offline MBRL often lack robustness during deployment, and small adversarial noise in the environment can lead to significant performance degradation. To address these, we propose a framework that dynamically adapts the world model alongside the policy under a unified learning objective aimed at improving robustness. At the core of our method is a maximin optimization problem, which we solve by innovatively utilizing Stackelberg learning dynamics. We provide theoretical analysis to support our design and introduce computationally efficient implementations. We benchmark our algorithm on twelve noisy D4RL MuJoCo tasks and three stochastic Tokamak Control tasks, demonstrating its state-of-the-art performance.

#### Research Highlights
- **Core Innovation:** We provide theoretical analysis to support our design and introduce computationally efficient implementations.
- **Methodology:** To address these, we propose a framework that dynamically adapts the world model alongside the policy under a unified learning objective aimed at improving robustness.
- **Key Finding:** This objective mismatch results in a world model that is not necessarily optimized for effective policy learning.

#### Technical Context
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
* **Limits:** However, most existing offline MBRL methods follow a two-stage training procedure: first learning a world model by maximizing the likelihood of the observed transitions, then optimizing a policy to maximize its expected return under the learned model.
* **Signal Tags:** #ai #research

---


### Modeling and Topology Estimation of Low Rank Dynamical Networks
**Date:** 2025-11-11 | **Arxiv:** [2511.06674](https://arxiv.org/abs/2511.06674)

#### Abstract
Conventional topology learning methods for dynamical networks become inapplicable to processes exhibiting low-rank characteristics. To address this, we propose the low rank dynamical network model which ensures identifiability. By employing causal Wiener filtering, we establish a necessary and sufficient condition that links the sparsity pattern of the filter to conditional Granger causality. Building on this theoretical result, we develop a consistent method for estimating all network edges. Simulation results demonstrate the parsimony of the proposed framework and consistency of the topology estimation approach.

#### Research Highlights
- **Core Innovation:** Simulation results demonstrate the parsimony of the proposed framework and consistency of the topology estimation approach..
- **Methodology:** Simulation results demonstrate the parsimony of the proposed framework and consistency of the topology estimation approach..
- **Key Finding:** Simulation results demonstrate the parsimony of the proposed framework and consistency of the topology estimation approach..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Decoupling Augmentation Bias in Prompt Learning for Vision-Language Models
**Date:** 2025-11-06 | **Arxiv:** [2511.03367](https://arxiv.org/abs/2511.03367)

#### Abstract
Recent advances in large-scale vision and language models have led to significant progress in zero-shot learning tasks. Methods such as CoOp and CoCoOp have shown that replacing handcrafted prompts with learnable vectors, known as prompt learning, can result in improved performance. However, these models often struggle to generalize to entirely unseen categories. While traditional zero-shot learning techniques benefit from various data augmentation strategies, prompt learning has primarily focused on text-based modifications, leaving the potential of image-based augmentation largely unexplored. In this work, we explore how image-level augmentations, particularly those that introduce attribute-specific variations, can support and enhance prompt learning. Our analysis examines the interaction between these augmentations and soft prompt frameworks, revealing their potential to improve generalization. We also identify a limitation in existing methods, such as CoCoOp, which do not provide explicit guidance for learning prompts that focus on semantically meaningful visual features. To address this, we propose Adding Attributes to Prompt Learning, AAPL, a novel method that introduces adversarial token embeddings to decouple superficial visual variations introduced by augmentation from class-relevant semantic representations. This decoupling enables the learned prompts to concentrate on visually discriminative features that align with the target categories. We conduct comprehensive experiments on eleven benchmark datasets, and AAPL consistently outperforms existing methods across few-shot, zero-shot, cross-dataset, and domain generalization settings. Our source code is publicly available at: https://github.com/Gahyeonkim09/AAPL

#### Research Highlights
- **Core Innovation:** To address this, we propose Adding Attributes to Prompt Learning, AAPL, a novel method that introduces adversarial token embeddings to decouple superficial visual variations introduced by augmentation from class-relevant semantic representations.
- **Methodology:** Our analysis examines the interaction between these augmentations and soft prompt frameworks, revealing their potential to improve generalization.
- **Key Finding:** Methods such as CoOp and CoCoOp have shown that replacing handcrafted prompts with learnable vectors, known as prompt learning, can result in improved performance.

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** AI Safety
* **Layer:** Application
* **Limits:** However, these models often struggle to generalize to entirely unseen categories.
* **Signal Tags:** #ai #research

---


### Learning Interactive World Model for Object-Centric Reinforcement Learning
**Date:** 2025-11-05 | **Arxiv:** [2511.02225](https://arxiv.org/abs/2511.02225)

#### Abstract
Agents that understand objects and their interactions can learn policies that are more robust and transferable. However, most object-centric RL methods factor state by individual objects while leaving interactions implicit. We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model. FIOC-WM captures environment dynamics with disentangled and modular representations of object interactions, improving sample efficiency and generalization for policy learning. Concretely, FIOC-WM first learns object-centric latents and an interaction structure directly from pixels, leveraging pre-trained vision encoders. The learned world model then decomposes tasks into composable interaction primitives, and a hierarchical policy is trained on top: a high level selects the type and order of interactions, while a low level executes them. On simulated robotic and embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and generalization over world-model baselines, indicating that explicit, modular interaction learning is crucial for robust control.

#### Research Highlights
- **Core Innovation:** We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model.
- **Methodology:** We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model.
- **Key Finding:** On simulated robotic and embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and generalization over world-model baselines, indicating that explicit, modular interaction learning is crucial for robust control..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Application
* **Limits:** However, most object-centric RL methods factor state by individual objects while leaving interactions implicit.
* **Signal Tags:** #ai #research

---


### DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning
**Date:** 2025-11-05 | **Arxiv:** [2505.20241](https://arxiv.org/abs/2505.20241)

#### Abstract
Reasoning has substantially improved the performance of large language models (LLMs) on complicated tasks. Central to the current reasoning studies, Process Reward Models (PRMs) offer a fine-grained evaluation of intermediate reasoning steps and guide the reasoning process. However, extending PRMs to multimodal large language models (MLLMs) introduces challenges. Since multimodal reasoning covers a wider range of tasks compared to text-only scenarios, the resulting distribution shift from the training to testing sets is more severe, leading to greater generalization difficulty. Training a reliable multimodal PRM, therefore, demands large and diverse datasets to ensure sufficient coverage. However, current multimodal reasoning datasets suffer from a marked quality imbalance, which degrades PRM performance and highlights the need for an effective data selection strategy. To address the issues, we introduce DreamPRM, a domain-reweighted training framework for multimodal PRMs which employs bi-level optimization. In the lower-level optimization, DreamPRM performs fine-tuning on multiple datasets with domain weights, allowing the PRM to prioritize high-quality reasoning signals and alleviating the impact of dataset quality imbalance. In the upper-level optimization, the PRM is evaluated on a separate meta-learning dataset; this feedback updates the domain weights through an aggregation loss function, thereby improving the generalization capability of trained PRM. Extensive experiments on multiple multimodal reasoning benchmarks covering both mathematical and general reasoning show that test-time scaling with DreamPRM consistently improves the performance of state-of-the-art MLLMs. Further comparisons reveal that DreamPRM's domain-reweighting strategy surpasses other data selection methods and yields higher accuracy gains than existing test-time scaling approaches.

#### Research Highlights
- **Core Innovation:** To address the issues, we introduce DreamPRM, a domain-reweighted training framework for multimodal PRMs which employs bi-level optimization.
- **Methodology:** In the lower-level optimization, DreamPRM performs fine-tuning on multiple datasets with domain weights, allowing the PRM to prioritize high-quality reasoning signals and alleviating the impact of dataset quality imbalance.
- **Key Finding:** Extensive experiments on multiple multimodal reasoning benchmarks covering both mathematical and general reasoning show that test-time scaling with DreamPRM consistently improves the performance of state-of-the-art MLLMs.

#### Technical Context
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
* **Limits:** However, extending PRMs to multimodal large language models (MLLMs) introduces challenges.
* **Signal Tags:** #ai #research

---


### Gaining Momentum: Uncovering Hidden Scoring Dynamics in Hockey through Deep Neural Sequencing and Causal Modeling
**Date:** 2025-11-04 | **Arxiv:** [2511.00615](https://arxiv.org/abs/2511.00615)

#### Abstract
We present a unified, data-driven framework for quantifying and enhancing offensive momentum and scoring likelihood (expected goals, xG) in professional hockey. Leveraging a Sportlogiq dataset of 541,000 NHL event records, our end-to-end pipeline comprises five stages: (1) interpretable momentum weighting of micro-events via logistic regression; (2) nonlinear xG estimation using gradient-boosted decision trees; (3) temporal sequence modeling with Long Short-Term Memory (LSTM) networks; (4) spatial formation discovery through principal component analysis (PCA) followed by K-Means clustering on standardized player coordinates; and (5) use of an X-Learner causal inference estimator to quantify the average treatment effect (ATE) of adopting the identified "optimal" event sequences and formations. We observe an ATE of 0.12 (95% CI: 0.05-0.17, p < 1e-50), corresponding to a 15% relative gain in scoring potential. These results demonstrate that strategically structured sequences and compact formations causally elevate offensive performance. Our framework delivers real-time, actionable insights for coaches and analysts, advancing hockey analytics toward principled, causally grounded tactical optimization.

#### Research Highlights
- **Core Innovation:** We present a unified, data-driven framework for quantifying and enhancing offensive momentum and scoring likelihood (expected goals, xG) in professional hockey.
- **Methodology:** Our framework delivers real-time, actionable insights for coaches and analysts, advancing hockey analytics toward principled, causally grounded tactical optimization..
- **Key Finding:** These results demonstrate that strategically structured sequences and compact formations causally elevate offensive performance.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Using machine learning methods to predict cognitive age from psychophysiological tests
**Date:** 2025-11-04 | **Arxiv:** [2511.00013](https://arxiv.org/abs/2511.00013)

#### Abstract
This study introduces a novel method for predicting cognitive age using psychophysiological tests. To determine cognitive age, subjects were asked to complete a series of psychological tests measuring various cognitive functions, including reaction time and cognitive conflict, short-term memory, verbal functions, and color and spatial perception. Based on the tests completed, the average completion time, proportion of correct answers, average absolute delta of the color campimetry test, number of guessed words in the Münsterberg matrix, and other parameters were calculated for each subject. The obtained characteristics of the subjects were preprocessed and used to train a machine learning algorithm implementing a regression task for predicting a person's cognitive age. These findings contribute to the field of remote screening using mobile devices for human health for diagnosing and monitoring cognitive aging.

#### Research Highlights
- **Core Innovation:** This study introduces a novel method for predicting cognitive age using psychophysiological tests.
- **Methodology:** These findings contribute to the field of remote screening using mobile devices for human health for diagnosing and monitoring cognitive aging..
- **Key Finding:** These findings contribute to the field of remote screening using mobile devices for human health for diagnosing and monitoring cognitive aging..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Towards Automated Semantic Interpretability in Reinforcement Learning via Vision-Language Models
**Date:** 2025-11-03 | **Arxiv:** [2503.16724](https://arxiv.org/abs/2503.16724)

#### Abstract
Semantic interpretability in Reinforcement Learning (RL) enables transparency and verifiability of decision-making. Achieving semantic interpretability in reinforcement learning requires (1) a feature space composed of human-understandable concepts and (2) a policy that is interpretable and verifiable. However, constructing such a feature space has traditionally relied on manual human specification, which often fails to generalize to unseen environments. Moreover, even when interpretable features are available, most reinforcement learning algorithms employ black-box models as policies, thereby hindering transparency. We introduce interpretable Tree-based Reinforcement learning via Automated Concept Extraction (iTRACE), an automated framework that leverages pre-trained vision-language models (VLM) for semantic feature extraction and train a interpretable tree-based model via RL. To address the impracticality of running VLMs in RL loops, we distill their outputs into a lightweight model. By leveraging Vision-Language Models (VLMs) to automate tree-based reinforcement learning, iTRACE loosens the reliance the need for human annotation that is traditionally required by interpretable models. In addition, it addresses key limitations of VLMs alone, such as their lack of grounding in action spaces and their inability to directly optimize policies. We evaluate iTRACE across three domains: Atari games, grid-world navigation, and driving. The results show that iTRACE outperforms other interpretable policy baselines and matches the performance of black-box policies on the same interpretable feature space.

#### Research Highlights
- **Core Innovation:** We introduce interpretable Tree-based Reinforcement learning via Automated Concept Extraction (iTRACE), an automated framework that leverages pre-trained vision-language models (VLM) for semantic feature extraction and train a interpretable tree-based model via RL.
- **Methodology:** We introduce interpretable Tree-based Reinforcement learning via Automated Concept Extraction (iTRACE), an automated framework that leverages pre-trained vision-language models (VLM) for semantic feature extraction and train a interpretable tree-based model via RL.
- **Key Finding:** The results show that iTRACE outperforms other interpretable policy baselines and matches the performance of black-box policies on the same interpretable feature space..

#### Technical Context
**Domain:** AI Research
**Approach:** Deep Learning
**Scale:** Unspecified

#### Reproducibility & Resources
* **Code:** Likely Available (See paper)
* **Data:** Not Mentioned
* **Benchmark:** Not Mentioned

---

**Metadata (Derived)**
* **Construct:** Autonomous Agent
* **Layer:** Infrastructure
* **Limits:** However, constructing such a feature space has traditionally relied on manual human specification, which often fails to generalize to unseen environments.
* **Signal Tags:** #ai #research

---


### Learning Geometry: A Framework for Building Adaptive Manifold Models through Metric Optimization
**Date:** 2025-10-31 | **Arxiv:** [2510.26068](https://arxiv.org/abs/2510.26068)

#### Abstract
This paper proposes a novel paradigm for machine learning that moves beyond traditional parameter optimization. Unlike conventional approaches that search for optimal parameters within a fixed geometric space, our core idea is to treat the model itself as a malleable geometric entity. Specifically, we optimize the metric tensor field on a manifold with a predefined topology, thereby dynamically shaping the geometric structure of the model space. To achieve this, we construct a variational framework whose loss function carefully balances data fidelity against the intrinsic geometric complexity of the manifold. The former ensures the model effectively explains observed data, while the latter acts as a regularizer, penalizing overly curved or irregular geometries to encourage simpler models and prevent overfitting. To address the computational challenges of this infinite-dimensional optimization problem, we introduce a practical method based on discrete differential geometry: the continuous manifold is discretized into a triangular mesh, and the metric tensor is parameterized by edge lengths, enabling efficient optimization using automatic differentiation tools. Theoretical analysis reveals a profound analogy between our framework and the Einstein-Hilbert action in general relativity, providing an elegant physical interpretation for the concept of "data-driven geometry". We further argue that even with fixed topology, metric optimization offers significantly greater expressive power than models with fixed geometry. This work lays a solid foundation for constructing fully dynamic "meta-learners" capable of autonomously evolving their geometry and topology, and it points to broad application prospects in areas such as scientific model discovery and robust representation learning.

#### Research Highlights
- **Core Innovation:** To address the computational challenges of this infinite-dimensional optimization problem, we introduce a practical method based on discrete differential geometry: the continuous manifold is discretized into a triangular mesh, and the metric tensor is parameterized by edge lengths, enabling efficient optimization using automatic differentiation tools.
- **Methodology:** Theoretical analysis reveals a profound analogy between our framework and the Einstein-Hilbert action in general relativity, providing an elegant physical interpretation for the concept of "data-driven geometry".
- **Key Finding:** This work lays a solid foundation for constructing fully dynamic "meta-learners" capable of autonomously evolving their geometry and topology, and it points to broad application prospects in areas such as scientific model discovery and robust representation learning..

#### Technical Context
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
* **Limits:** challenges of this infinite-dimensional optimization problem, we introduce a practical method based on discrete differential geometry: the continuous manifold is discretized into a triangular mesh, and the metric tensor is parameterized by edge lengths, enabling efficient optimization using automatic differentiation tools.
* **Signal Tags:** #ai #research

---


### SteerVLM: Robust Model Control through Lightweight Activation Steering for Vision Language Models
**Date:** 2025-10-31 | **Arxiv:** [2510.26769](https://arxiv.org/abs/2510.26769)

#### Abstract
This work introduces SteerVLM, a lightweight steering module designed to guide Vision-Language Models (VLMs) towards outputs that better adhere to desired instructions. Our approach learns from the latent embeddings of paired prompts encoding target and converse behaviors to dynamically adjust activations connecting the language modality with image context. This allows for fine-grained, inference-time control over complex output semantics without modifying model weights while preserving performance on off-target tasks. Our steering module requires learning parameters equal to 0.14% of the original VLM's size. Our steering module gains model control through dimension-wise activation modulation and adaptive steering across layers without requiring pre-extracted static vectors or manual tuning of intervention points. Furthermore, we introduce VNIA (Visual Narrative Intent Alignment), a multimodal dataset specifically created to facilitate the development and evaluation of VLM steering techniques. Our method outperforms existing intervention techniques on steering and hallucination mitigation benchmarks for VLMs and proposes a robust solution for multimodal model control through activation engineering.

#### Research Highlights
- **Core Innovation:** Our method outperforms existing intervention techniques on steering and hallucination mitigation benchmarks for VLMs and proposes a robust solution for multimodal model control through activation engineering..
- **Methodology:** See abstract.
- **Key Finding:** Our method outperforms existing intervention techniques on steering and hallucination mitigation benchmarks for VLMs and proposes a robust solution for multimodal model control through activation engineering..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Revisiting Service Level Objectives and System Level Metrics in Large Language Model Serving
**Date:** 2025-10-30 | **Arxiv:** [2410.14257](https://arxiv.org/abs/2410.14257)

#### Abstract
User experience is a critical factor Large Language Model (LLM) serving systems must consider, where service level objectives (SLOs) considering the experience of individual requests and system level metrics (SLMs) considering the overall system performance are two key performance measures. However, we observe two notable issues in existing metrics: 1) manually delaying the delivery of some tokens can improve SLOs, and 2) actively abandoning requests that do not meet SLOs can improve SLMs, both of which are counterintuitive.   In this paper, we revisit SLOs and SLMs in LLM serving, and propose a new SLO that aligns with user experience. Based on the SLO, we propose a comprehensive metric framework called smooth goodput, which integrates SLOs and SLMs to reflect the nature of user experience in LLM serving. Through this unified framework, we reassess the performance of different LLM serving systems under multiple workloads. Evaluation results show that our metric framework provides a more comprehensive view of token delivery and request processing, and effectively captures the optimal point of user experience and system performance with different serving strategies.

#### Research Highlights
- **Core Innovation:** Based on the SLO, we propose a comprehensive metric framework called smooth goodput, which integrates SLOs and SLMs to reflect the nature of user experience in LLM serving.
- **Methodology:** Evaluation results show that our metric framework provides a more comprehensive view of token delivery and request processing, and effectively captures the optimal point of user experience and system performance with different serving strategies..
- **Key Finding:** Evaluation results show that our metric framework provides a more comprehensive view of token delivery and request processing, and effectively captures the optimal point of user experience and system performance with different serving strategies..

#### Technical Context
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
* **Limits:** However, we observe two notable issues in existing metrics: 1) manually delaying the delivery of some tokens can improve SLOs, and 2) actively abandoning requests that do not meet SLOs can improve SLMs, both of which are counterintuitive.
* **Signal Tags:** #ai #research

---


### RAVR: Reference-Answer-guided Variational Reasoning for Large Language Models
**Date:** 2025-10-30 | **Arxiv:** [2510.25206](https://arxiv.org/abs/2510.25206)

#### Abstract
Reinforcement learning (RL) can refine the reasoning abilities of large language models (LLMs), but critically depends on a key prerequisite: the LLM can already generate high-utility reasoning paths with non-negligible probability. For tasks beyond the LLM's current competence, such reasoning path can be hard to sample, and learning risks reinforcing familiar but suboptimal reasoning. We are motivated by the insight from cognitive science that Why is this the answer is often an easier question than What is the answer, as it avoids the heavy cognitive load of open-ended exploration, opting instead for explanatory reconstruction-systematically retracing the reasoning that links a question to its answer. We show that LLMs can similarly leverage answers to derive high-quality reasoning paths. We formalize this phenomenon and prove that conditioning on answer provably increases the expected utility of sampled reasoning paths, thereby transforming intractable problems into learnable ones. Building on this insight, we introduce RAVR (Reference-Answer-guided Variational Reasoning), an end-to-end framework that uses answer-conditioned reasoning as a variational surrogate for question-only reasoning. Experiments in both general and math domains demonstrate consistent improvements over strong baselines. We further analyze the reasoning behavior and find that RAVR reduces hesitation, strengthens conclusion consolidation, and promotes problem-specific strategies in reasoning.

#### Research Highlights
- **Core Innovation:** Building on this insight, we introduce RAVR (Reference-Answer-guided Variational Reasoning), an end-to-end framework that uses answer-conditioned reasoning as a variational surrogate for question-only reasoning.
- **Methodology:** Building on this insight, we introduce RAVR (Reference-Answer-guided Variational Reasoning), an end-to-end framework that uses answer-conditioned reasoning as a variational surrogate for question-only reasoning.
- **Key Finding:** Experiments in both general and math domains demonstrate consistent improvements over strong baselines.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---
