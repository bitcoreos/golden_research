# Vol 02 General Architecture Research
*Enriched by BITCOREOS | Phase 4 Batch 1*

---

### DeCaFlow: A deconfounding causal generative model
**Date:** 2025-10-27 | **Arxiv:** [2503.15114](https://arxiv.org/abs/2503.15114)

#### Abstract
We introduce DeCaFlow, a deconfounding causal generative model. Training once per dataset using just observational data and the underlying causal graph, DeCaFlow enables accurate causal inference on continuous variables under the presence of hidden confounders. Specifically, we extend previous results on causal estimation under hidden confounding to show that a single instance of DeCaFlow provides correct estimates for all causal queries identifiable with do-calculus, leveraging proxy variables to adjust for the causal effects when do-calculus alone is insufficient. Moreover, we show that counterfactual queries are identifiable as long as their interventional counterparts are identifiable, and thus are also correctly estimated by DeCaFlow. Our empirical results on diverse settings (including the Ecoli70 dataset, with 3 independent hidden confounders, tens of observed variables and hundreds of causal queries) show that DeCaFlow outperforms existing approaches, while demonstrating its out-of-the-box applicability to any given causal graph. An implementation can be found in https://github.com/aalmodovares/DeCaFlow

#### Research Highlights
- **Core Innovation:** We introduce DeCaFlow, a deconfounding causal generative model.
- **Methodology:** Training once per dataset using just observational data and the underlying causal graph, DeCaFlow enables accurate causal inference on continuous variables under the presence of hidden confounders.
- **Key Finding:** Our empirical results on diverse settings (including the Ecoli70 dataset, with 3 independent hidden confounders, tens of observed variables and hundreds of causal queries) show that DeCaFlow outperforms existing approaches, while demonstrating its out-of-the-box applicability to any given causal graph.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Towards Machine Learning-based Model Predictive Control for HVAC Control in Multi-Context Buildings at Scale via Ensemble Learning
**Date:** 2025-10-24 | **Arxiv:** [2505.02439](https://arxiv.org/abs/2505.02439)

#### Abstract
The building thermodynamics model, which predicts real-time indoor temperature changes under potential HVAC (Heating, Ventilation, and Air Conditioning) control operations, is crucial for optimizing HVAC control in buildings. While pioneering studies have attempted to develop such models for various building environments, these models often require extensive data collection periods and rely heavily on expert knowledge, making the modeling process inefficient and limiting the reusability of the models. This paper explores a model ensemble perspective that utilizes existing developed models as base models to serve a target building environment, thereby providing accurate predictions while reducing the associated efforts. Given that building data streams are non-stationary and the number of base models may increase, we propose a Hierarchical Reinforcement Learning (HRL) approach to dynamically select and weight the base models. Our approach employs a two-tiered decision-making process: the high-level focuses on model selection, while the low-level determines the weights of the selected models. We thoroughly evaluate the proposed approach through offline experiments and an on-site case study, and the experimental results demonstrate the effectiveness of our method.

#### Research Highlights
- **Core Innovation:** We thoroughly evaluate the proposed approach through offline experiments and an on-site case study, and the experimental results demonstrate the effectiveness of our method..
- **Methodology:** See abstract.
- **Key Finding:** We thoroughly evaluate the proposed approach through offline experiments and an on-site case study, and the experimental results demonstrate the effectiveness of our method..

#### Technical Context
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


### VAMOS: A Hierarchical Vision-Language-Action Model for Capability-Modulated and Steerable Navigation
**Date:** 2025-10-24 | **Arxiv:** [2510.20818](https://arxiv.org/abs/2510.20818)

#### Abstract
A fundamental challenge in robot navigation lies in learning policies that generalize across diverse environments while conforming to the unique physical constraints and capabilities of a specific embodiment (e.g., quadrupeds can walk up stairs, but rovers cannot). We propose VAMOS, a hierarchical VLA that decouples semantic planning from embodiment grounding: a generalist planner learns from diverse, open-world data, while a specialist affordance model learns the robot's physical constraints and capabilities in safe, low-cost simulation. We enabled this separation by carefully designing an interface that lets a high-level planner propose candidate paths directly in image space that the affordance model then evaluates and re-ranks. Our real-world experiments show that VAMOS achieves higher success rates in both indoor and complex outdoor navigation than state-of-the-art model-based and end-to-end learning methods. We also show that our hierarchical design enables cross-embodied navigation across legged and wheeled robots and is easily steerable using natural language. Real-world ablations confirm that the specialist model is key to embodiment grounding, enabling a single high-level planner to be deployed across physically distinct wheeled and legged robots. Finally, this model significantly enhances single-robot reliability, achieving 3X higher success rates by rejecting physically infeasible plans. Website: https://vamos-vla.github.io/

#### Research Highlights
- **Core Innovation:** We enabled this separation by carefully designing an interface that lets a high-level planner propose candidate paths directly in image space that the affordance model then evaluates and re-ranks.
- **Methodology:** We also show that our hierarchical design enables cross-embodied navigation across legged and wheeled robots and is easily steerable using natural language.
- **Key Finding:** We also show that our hierarchical design enables cross-embodied navigation across legged and wheeled robots and is easily steerable using natural language.

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
* **Limits:** challenge in robot navigation lies in learning policies that generalize across diverse environments while conforming to the unique physical constraints and capabilities of a specific embodiment (e.
* **Signal Tags:** #ai #research

---


### Model-based Large Language Model Customization as Service
**Date:** 2025-10-23 | **Arxiv:** [2410.10481](https://arxiv.org/abs/2410.10481)

#### Abstract
Prominent Large Language Model (LLM) services from providers like OpenAI and Google excel at general tasks but often underperform on domain-specific applications. Current customization services for these LLMs typically require users to upload data for fine-tuning, posing significant privacy risks. While differentially private (DP) data synthesis presents a potential alternative, its application commonly results in low effectiveness due to the introduction of excessive noise on data for DP. To overcome this, we introduce Llamdex, a novel framework that facilitates LLM customization as a service, where the client uploads pre-trained domain-specific models rather than data. This client-uploaded model, optionally protected by DP with much lower noise, is inserted into the base LLM via connection modules. Significantly, these connecting modules are trained without requiring sensitive domain data, enabling clients to customize LLM services while preserving data privacy. Experiments demonstrate that Llamdex improves domain-specific accuracy by up to 26% over state-of-the-art private data synthesis methods under identical privacy constraints and, by obviating the need for users to provide domain context within queries, maintains inference efficiency comparable to the original LLM service.

#### Research Highlights
- **Core Innovation:** To overcome this, we introduce Llamdex, a novel framework that facilitates LLM customization as a service, where the client uploads pre-trained domain-specific models rather than data.
- **Methodology:** Experiments demonstrate that Llamdex improves domain-specific accuracy by up to 26% over state-of-the-art private data synthesis methods under identical privacy constraints and, by obviating the need for users to provide domain context within queries, maintains inference efficiency comparable to the original LLM service..
- **Key Finding:** Experiments demonstrate that Llamdex improves domain-specific accuracy by up to 26% over state-of-the-art private data synthesis methods under identical privacy constraints and, by obviating the need for users to provide domain context within queries, maintains inference efficiency comparable to the original LLM service..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Preliminary Use of Vision Language Model Driven Extraction of Mouse Behavior Towards Understanding Fear Expression
**Date:** 2025-10-23 | **Arxiv:** [2510.19160](https://arxiv.org/abs/2510.19160)

#### Abstract
Integration of diverse data will be a pivotal step towards improving scientific explorations in many disciplines. This work establishes a vision-language model (VLM) that encodes videos with text input in order to classify various behaviors of a mouse existing in and engaging with their environment. Importantly, this model produces a behavioral vector over time for each subject and for each session the subject undergoes. The output is a valuable dataset that few programs are able to produce with as high accuracy and with minimal user input. Specifically, we use the open-source Qwen2.5-VL model and enhance its performance through prompts, in-context learning (ICL) with labeled examples, and frame-level preprocessing. We found that each of these methods contributes to improved classification, and that combining them results in strong F1 scores across all behaviors, including rare classes like freezing and fleeing, without any model fine-tuning. Overall, this model will support interdisciplinary researchers studying mouse behavior by enabling them to integrate diverse behavioral features, measured across multiple time points and environments, into a comprehensive dataset that can address complex research questions.

#### Research Highlights
- **Core Innovation:** Integration of diverse data will be a pivotal step towards improving scientific explorations in many disciplines.
- **Methodology:** See abstract.
- **Key Finding:** We found that each of these methods contributes to improved classification, and that combining them results in strong F1 scores across all behaviors, including rare classes like freezing and fleeing, without any model fine-tuning.

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
* **Construct:** Machine Perception
* **Layer:** Theory
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Illusions of reflection: open-ended task reveals systematic failures in Large Language Models' reflective reasoning
**Date:** 2025-10-22 | **Arxiv:** [2510.18254](https://arxiv.org/abs/2510.18254)

#### Abstract
Humans do not just find mistakes after the fact -- we often catch them mid-stream because 'reflection' is tied to the goal and its constraints. Today's large language models produce reasoning tokens and 'reflective' text, but is it functionally equivalent with human reflective reasoning? Prior work on closed-ended tasks -- with clear, external 'correctness' signals -- can make 'reflection' look effective while masking limits in self-correction. We therefore test eight frontier models on a simple, real-world task that is open-ended yet rule-constrained, with auditable success criteria: to produce valid scientific test items, then revise after considering their own critique. First-pass performance is poor (often zero valid items out of 4 required; mean $\approx$ 1), and reflection yields only modest gains (also $\approx$ 1). Crucially, the second attempt frequently repeats the same violation of constraint, indicating 'corrective gains' arise largely from chance production of a valid item rather than error detection and principled, constraint-sensitive repair. Performance before and after reflection deteriorates as open-endedness increases, and models marketed for 'reasoning' show no advantage. Our results suggest that current LLM 'reflection' lacks functional evidence of the active, goal-driven monitoring that helps humans respect constraints even on a first pass. Until such mechanisms are instantiated in the model itself, reliable performance requires external structure that enforces constraints. Our code is available at: https://github.com/cruiseresearchgroup/LLM_ReflectionTest

#### Research Highlights
- **Core Innovation:** Humans do not just find mistakes after the fact -- we often catch them mid-stream because 'reflection' is tied to the goal and its constraints.
- **Methodology:** See abstract.
- **Key Finding:** Our results suggest that current LLM 'reflection' lacks functional evidence of the active, goal-driven monitoring that helps humans respect constraints even on a first pass.

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


### From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors
**Date:** 2025-10-21 | **Arxiv:** [2510.17439](https://arxiv.org/abs/2510.17439)

#### Abstract
Existing vision-language-action (VLA) models act in 3D real-world but are typically built on 2D encoders, leaving a spatial reasoning gap that limits generalization and adaptability. Recent 3D integration techniques for VLAs either require specialized sensors and transfer poorly across modalities, or inject weak cues that lack geometry and degrade vision-language alignment. In this work, we introduce FALCON (From Spatial to Action), a novel paradigm that injects rich 3D spatial tokens into the action head. FALCON leverages spatial foundation models to deliver strong geometric priors from RGB alone, and includes an Embodied Spatial Model that can optionally fuse depth, or pose for higher fidelity when available, without retraining or architectural changes. To preserve language reasoning, spatial tokens are consumed by a Spatial-Enhanced Action Head rather than being concatenated into the vision-language backbone. These designs enable FALCON to address limitations in spatial representation, modality transferability, and alignment. In comprehensive evaluations across three simulation benchmarks and eleven real-world tasks, our proposed FALCON achieves state-of-the-art performance, consistently surpasses competitive baselines, and remains robust under clutter, spatial-prompt conditioning, and variations in object scale and height.

#### Research Highlights
- **Core Innovation:** In comprehensive evaluations across three simulation benchmarks and eleven real-world tasks, our proposed FALCON achieves state-of-the-art performance, consistently surpasses competitive baselines, and remains robust under clutter, spatial-prompt conditioning, and variations in object scale and height..
- **Methodology:** See abstract.
- **Key Finding:** In comprehensive evaluations across three simulation benchmarks and eleven real-world tasks, our proposed FALCON achieves state-of-the-art performance, consistently surpasses competitive baselines, and remains robust under clutter, spatial-prompt conditioning, and variations in object scale and height..

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
* **Limits:** limitations in spatial representation, modality transferability, and alignment.
* **Signal Tags:** #ai #research

---


### FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models
**Date:** 2025-10-21 | **Arxiv:** [2508.01055](https://arxiv.org/abs/2508.01055)

#### Abstract
Large language models (LLMs) have gained significant attention in chemistry. However, most existing datasets center on molecular-level property prediction and overlook the role of fine-grained functional group (FG) information. Incorporating FG-level data can provide valuable prior knowledge that links molecular structures with textual descriptions, which can be used to build more interpretable, structure-aware LLMs for reasoning on molecule-related tasks. Moreover, LLMs can learn from such fine-grained information to uncover hidden relationships between specific functional groups and molecular properties, thereby advancing molecular design and drug discovery. Here, we introduce FGBench, a dataset comprising 625K molecular property reasoning problems with functional group information. Functional groups are precisely annotated and localized within the molecule, which ensures the dataset's interoperability thereby facilitating further multimodal applications. FGBench includes both regression and classification tasks on 245 different functional groups across three categories for molecular property reasoning: (1) single functional group impacts, (2) multiple functional group interactions, and (3) direct molecular comparisons. In the benchmark of state-of-the-art LLMs on 7K curated data, the results indicate that current LLMs struggle with FG-level property reasoning, highlighting the need to enhance reasoning capabilities in LLMs for chemistry tasks. We anticipate that the methodology employed in FGBench to construct datasets with functional group-level information will serve as a foundational framework for generating new question-answer pairs, enabling LLMs to better understand fine-grained molecular structure-property relationships. The dataset and evaluation code are available at https://github.com/xuanliugit/FGBench.

#### Research Highlights
- **Core Innovation:** Here, we introduce FGBench, a dataset comprising 625K molecular property reasoning problems with functional group information.
- **Methodology:** We anticipate that the methodology employed in FGBench to construct datasets with functional group-level information will serve as a foundational framework for generating new question-answer pairs, enabling LLMs to better understand fine-grained molecular structure-property relationships.
- **Key Finding:** In the benchmark of state-of-the-art LLMs on 7K curated data, the results indicate that current LLMs struggle with FG-level property reasoning, highlighting the need to enhance reasoning capabilities in LLMs for chemistry tasks.

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
* **Limits:** However, most existing datasets center on molecular-level property prediction and overlook the role of fine-grained functional group (FG) information.
* **Signal Tags:** #ai #research

---


### Algorithmic Primitives and Compositional Geometry of Reasoning in Language Models
**Date:** 2025-10-21 | **Arxiv:** [2510.15987](https://arxiv.org/abs/2510.15987)

#### Abstract
How do latent and inference time computations enable large language models (LLMs) to solve multi-step reasoning? We introduce a framework for tracing and steering algorithmic primitives that underlie model reasoning. Our approach links reasoning traces to internal activation patterns and evaluates algorithmic primitives by injecting them into residual streams and measuring their effect on reasoning steps and task performance. We consider four benchmarks: Traveling Salesperson Problem (TSP), 3SAT, AIME, and graph navigation. We operationalize primitives by clustering neural activations and labeling their matched reasoning traces. We then apply function vector methods to derive primitive vectors as reusable compositional building blocks of reasoning. Primitive vectors can be combined through addition, subtraction, and scalar operations, revealing a geometric logic in activation space. Cross-task and cross-model evaluations (Phi-4, Phi-4-Reasoning, Llama-3-8B) show both shared and task-specific primitives. Notably, comparing Phi-4 with its reasoning-finetuned variant highlights compositional generalization after finetuning: Phi-4-Reasoning exhibits more systematic use of verification and path-generation primitives. Injecting the associated primitive vectors in Phi-4-Base induces behavioral hallmarks associated with Phi-4-Reasoning. Together, these findings demonstrate that reasoning in LLMs may be supported by a compositional geometry of algorithmic primitives, that primitives transfer cross-task and cross-model, and that reasoning finetuning strengthens algorithmic generalization across domains.

#### Research Highlights
- **Core Innovation:** How do latent and inference time computations enable large language models (LLMs) to solve multi-step reasoning? We introduce a framework for tracing and steering algorithmic primitives that underlie model reasoning.
- **Methodology:** How do latent and inference time computations enable large language models (LLMs) to solve multi-step reasoning? We introduce a framework for tracing and steering algorithmic primitives that underlie model reasoning.
- **Key Finding:** Together, these findings demonstrate that reasoning in LLMs may be supported by a compositional geometry of algorithmic primitives, that primitives transfer cross-task and cross-model, and that reasoning finetuning strengthens algorithmic generalization across domains..

#### Technical Context
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


### Reasoning with Sampling: Your Base Model is Smarter Than You Think
**Date:** 2025-10-17 | **Arxiv:** [2510.14901](https://arxiv.org/abs/2510.14901)

#### Abstract
Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilites can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains.

#### Research Highlights
- **Core Innovation:** Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods.
- **Methodology:** See abstract.
- **Key Finding:** Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA.

#### Technical Context
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
* **Limits:** However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models.
* **Signal Tags:** #ai #research

---


### UniFusion: Vision-Language Model as Unified Encoder in Image Generation
**Date:** 2025-10-16 | **Arxiv:** [2510.12789](https://arxiv.org/abs/2510.12789)

#### Abstract
Although recent advances in visual generation have been remarkable, most existing architectures still depend on distinct encoders for images and text. This separation constrains diffusion models' ability to perform cross-modal reasoning and knowledge transfer. Prior attempts to bridge this gap often use the last layer information from VLM, employ multiple visual encoders, or train large unified models jointly for text and image generation, which demands substantial computational resources and large-scale data, limiting its accessibility.We present UniFusion, a diffusion-based generative model conditioned on a frozen large vision-language model (VLM) that serves as a unified multimodal encoder. At the core of UniFusion is the Layerwise Attention Pooling (LAP) mechanism that extracts both high level semantics and low level details from text and visual tokens of a frozen VLM to condition a diffusion generative model. We demonstrate that LAP outperforms other shallow fusion architectures on text-image alignment for generation and faithful transfer of visual information from VLM to the diffusion model which is key for editing. We propose VLM-Enabled Rewriting Injection with Flexibile Inference (VERIFI), which conditions a diffusion transformer (DiT) only on the text tokens generated by the VLM during in-model prompt rewriting. VERIFI combines the alignment of the conditioning distribution with the VLM's reasoning capabilities for increased capabilities and flexibility at inference. In addition, finetuning on editing task not only improves text-image alignment for generation, indicative of cross-modality knowledge transfer, but also exhibits tremendous generalization capabilities. Our model when trained on single image editing, zero-shot generalizes to multiple image references further motivating the unified encoder design of UniFusion.

#### Research Highlights
- **Core Innovation:** We propose VLM-Enabled Rewriting Injection with Flexibile Inference (VERIFI), which conditions a diffusion transformer (DiT) only on the text tokens generated by the VLM during in-model prompt rewriting.
- **Methodology:** See abstract.
- **Key Finding:** We demonstrate that LAP outperforms other shallow fusion architectures on text-image alignment for generation and faithful transfer of visual information from VLM to the diffusion model which is key for editing.

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
* **Limits:** Although recent advances in visual generation have been remarkable, most existing architectures still depend on distinct encoders for images and text.
* **Signal Tags:** #ai #research

---


### DeepCausalMMM: A Deep Learning Framework for Marketing Mix Modeling with Causal Inference
**Date:** 2025-10-16 | **Arxiv:** [2510.13087](https://arxiv.org/abs/2510.13087)

#### Abstract
Marketing Mix Modeling (MMM) is a statistical technique used to estimate the impact of marketing activities on business outcomes such as sales, revenue, or customer visits. Traditional MMM approaches often rely on linear regression or Bayesian hierarchical models that assume independence between marketing channels and struggle to capture complex temporal dynamics and non-linear saturation effects [@Chan2017; @Hanssens2005; @Ng2021Bayesian].   **DeepCausalMMM** is a Python package that addresses these limitations by combining deep learning, causal inference, and advanced marketing science. The package uses Gated Recurrent Units (GRUs) to automatically learn temporal patterns such as adstock (carryover effects) and lag, while simultaneously learning statistical dependencies and potential causal structures between marketing channels through Directed Acyclic Graph (DAG) learning [@Zheng2018NOTEARS; @Gong2024CausalMMM]. Additionally, it implements Hill equation-based saturation curves to model diminishing returns and optimize budget allocation.   Key features include: (1) a data-driven design where hyperparameters and transformations (e.g., adstock decay, saturation curves) are learned or estimated from data with sensible defaults, rather than requiring fixed heuristics or manual specification, (2) multi-region modeling with both shared and region-specific parameters, (3) robust statistical methods including Huber loss and advanced regularization, (4) comprehensive response curve analysis for understanding channel saturation.

#### Research Highlights
- **Core Innovation:** Marketing Mix Modeling (MMM) is a statistical technique used to estimate the impact of marketing activities on business outcomes such as sales, revenue, or customer visits.
- **Methodology:** See abstract.
- **Key Finding:**   Key features include: (1) a data-driven design where hyperparameters and transformations (e.g., adstock decay, saturation curves) are learned or estimated from data with sensible defaults, rather than requiring fixed heuristics or manual specification, (2) multi-region modeling with both shared and region-specific parameters, (3) robust statistical methods including Huber loss and advanced regularization, (4) comprehensive response curve analysis for understanding channel saturation..

#### Technical Context
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
* **Limits:** limitations by combining deep learning, causal inference, and advanced marketing science.
* **Signal Tags:** #ai #research

---


### FLAMMABLE: A Multi-Model Federated Learning Framework with Multi-Model Engagement and Adaptive Batch Sizes
**Date:** 2025-10-15 | **Arxiv:** [2510.10380](https://arxiv.org/abs/2510.10380)

#### Abstract
Multi-Model Federated Learning (MMFL) is an emerging direction in Federated Learning (FL) where multiple models are trained in parallel, generally on various datasets. Optimizing the models' accuracies and training times in the MMFL setting requires adapting to data and system heterogeneity across clients as in single-model FL; these challenges are amplified in the MMFL setting due to additional heterogeneity across models. Neither existing solutions nor naïve extensions of single-model FL frameworks efficiently address these challenges. To bridge this gap, we propose FLAMMABLE, a comprehensive MMFL training framework. FLAMMABLE optimizes model training by intelligently adapting client batch sizes while engaging them to train multiple carefully chosen models, depending on their system capabilities, in each training round. To evaluate FLAMMABLE, we develop the first benchmark platform for the MMFL setting, which may enable future reproducible MMFL research. Extensive evaluations on multiple datasets and models show that FLAMMABLE boosts the MMFL time-to-accuracy performance by 1.1$\sim$10.0$\times$ while improving the final model accuracy by 1.3$\sim$5.4\% compared to several known baselines.

#### Research Highlights
- **Core Innovation:** To bridge this gap, we propose FLAMMABLE, a comprehensive MMFL training framework.
- **Methodology:** To bridge this gap, we propose FLAMMABLE, a comprehensive MMFL training framework.
- **Key Finding:** Extensive evaluations on multiple datasets and models show that FLAMMABLE boosts the MMFL time-to-accuracy performance by 1.1$\sim$10.0$\times$ while improving the final model accuracy by 1.3$\sim$5.4\% compared to several known baselines..

#### Technical Context
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
* **Limits:** challenges are amplified in the MMFL setting due to additional heterogeneity across models.
* **Signal Tags:** #ai #research

---


### Discovering and Reasoning of Causality in the Hidden World with Large Language Models
**Date:** 2025-10-15 | **Arxiv:** [2402.03941](https://arxiv.org/abs/2402.03941)

#### Abstract
Revealing hidden causal variables alongside the underlying causal mechanisms is essential to the development of science. Despite the progress in the past decades, existing practice in causal discovery (CD) heavily relies on high-quality measured variables, which are usually given by human experts. In fact, the lack of well-defined high-level variables behind unstructured data has been a longstanding roadblock to a broader real-world application of CD. This procedure can naturally benefit from an automated process that can suggest potential hidden variables in the system. Interestingly, Large language models (LLMs) are trained on massive observations of the world and have demonstrated great capability in processing unstructured data. To leverage the power of LLMs, we develop a new framework termed Causal representatiOn AssistanT (COAT) that incorporates the rich world knowledge of LLMs to propose useful measured variables for CD with respect to high-value target variables on their paired unstructured data. Instead of directly inferring causality with LLMs, COAT constructs feedback from intermediate CD results to LLMs to refine the proposed variables. Given the target variable and the paired unstructured data, we first develop COAT-MB that leverages the predictivity of the proposed variables to iteratively uncover the Markov Blanket of the target variable. Built upon COAT-MB, COAT-PAG further extends to uncover a more complete causal graph, i.e., Partial Ancestral Graph, by iterating over the target variables and actively seeking new high-level variables. Moreover, the reliable CD capabilities of COAT also extend the debiased causal inference to unstructured data by discovering an adjustment set. We establish theoretical guarantees for the CD results and verify their efficiency and reliability across realistic benchmarks and real-world case studies.

#### Research Highlights
- **Core Innovation:** Given the target variable and the paired unstructured data, we first develop COAT-MB that leverages the predictivity of the proposed variables to iteratively uncover the Markov Blanket of the target variable.
- **Methodology:** To leverage the power of LLMs, we develop a new framework termed Causal representatiOn AssistanT (COAT) that incorporates the rich world knowledge of LLMs to propose useful measured variables for CD with respect to high-value target variables on their paired unstructured data.
- **Key Finding:** We establish theoretical guarantees for the CD results and verify their efficiency and reliability across realistic benchmarks and real-world case studies..

#### Technical Context
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


### Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models
**Date:** 2025-10-15 | **Arxiv:** [2510.10278](https://arxiv.org/abs/2510.10278)

#### Abstract
Clinical reasoning in medicine is a hypothesis-driven process where physicians refine diagnoses from limited information through targeted history, physical examination, and diagnostic investigations. In contrast, current medical benchmarks for large language models (LLMs) primarily assess knowledge recall through single-turn questions, where complete clinical information is provided upfront. To address this gap, we introduce VivaBench, a multi-turn benchmark that evaluates sequential clinical reasoning in LLM agents. Our dataset consists of 1762 physician-curated clinical vignettes structured as interactive scenarios that simulate a (oral) examination in medical training, requiring agents to actively probe for relevant findings, select appropriate investigations, and synthesize information across multiple steps to reach a diagnosis. While current LLMs demonstrate competence in diagnosing conditions from well-described clinical presentations, their performance degrades significantly when required to navigate iterative diagnostic reasoning under uncertainty in our evaluation. Our analysis identified several failure modes that mirror common cognitive errors in clinical practice, including: (1) fixation on initial hypotheses, (2) inappropriate investigation ordering, (3) premature diagnostic closure, and (4) failing to screen for critical conditions. These patterns reveal fundamental limitations in how current LLMs reason and make decisions under uncertainty. Through VivaBench, we provide a standardized benchmark for evaluating conversational medical AI systems for real-world clinical decision support. Beyond medical applications, we contribute to the larger corpus of research on agentic AI by demonstrating how sequential reasoning trajectories can diverge in complex decision-making environments.

#### Research Highlights
- **Core Innovation:** To address this gap, we introduce VivaBench, a multi-turn benchmark that evaluates sequential clinical reasoning in LLM agents.
- **Methodology:** See abstract.
- **Key Finding:** While current LLMs demonstrate competence in diagnosing conditions from well-described clinical presentations, their performance degrades significantly when required to navigate iterative diagnostic reasoning under uncertainty in our evaluation.

#### Technical Context
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
* **Limits:** limitations in how current LLMs reason and make decisions under uncertainty.
* **Signal Tags:** #ai #research

---


### Reasoning-Enhanced Large Language Models for Molecular Property Prediction
**Date:** 2025-10-15 | **Arxiv:** [2510.10248](https://arxiv.org/abs/2510.10248)

#### Abstract
Molecular property prediction is crucial for drug discovery and materials science, yet existing approaches suffer from limited interpretability, poor cross-task generalization, and lack of chemical reasoning capabilities. Traditional machine learning models struggle with task transferability, while specialized molecular language models provide little insight into their decision-making processes. To address these limitations, we propose \textbf{MPPReasoner}, a multimodal large language model that incorporates chemical reasoning for molecular property prediction. Our approach, built upon Qwen2.5-VL-7B-Instruct, integrates molecular images with SMILES strings to enable comprehensive molecular understanding. We develop a two-stage training strategy: supervised fine-tuning (SFT) using 16,000 high-quality reasoning trajectories generated through expert knowledge and multiple teacher models, followed by Reinforcement Learning from Principle-Guided Rewards (RLPGR). RLPGR employs verifiable, rule-based rewards that systematically evaluate chemical principle application, molecular structure analysis, and logical consistency through computational verification. Extensive experiments across 8 datasets demonstrate significant performance improvements, with MPPReasoner outperforming the best baselines by 7.91\% and 4.53\% on in-distribution and out-of-distribution tasks respectively. MPPReasoner exhibits exceptional cross-task generalization and generates chemically sound reasoning paths that provide valuable insights into molecular property analysis, substantially enhancing both interpretability and practical utility for chemists. Code is available at https://anonymous.4open.science/r/MPPReasoner-12687.

#### Research Highlights
- **Core Innovation:** To address these limitations, we propose \textbf{MPPReasoner}, a multimodal large language model that incorporates chemical reasoning for molecular property prediction.
- **Methodology:** We develop a two-stage training strategy: supervised fine-tuning (SFT) using 16,000 high-quality reasoning trajectories generated through expert knowledge and multiple teacher models, followed by Reinforcement Learning from Principle-Guided Rewards (RLPGR).
- **Key Finding:** Extensive experiments across 8 datasets demonstrate significant performance improvements, with MPPReasoner outperforming the best baselines by 7.91\% and 4.53\% on in-distribution and out-of-distribution tasks respectively.

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
* **Limits:** limitations, we propose \textbf{MPPReasoner}, a multimodal large language model that incorporates chemical reasoning for molecular property prediction.
* **Signal Tags:** #ai #research

---


### Language Model Guided Reinforcement Learning in Quantitative Trading
**Date:** 2025-10-13 | **Arxiv:** [2508.02366](https://arxiv.org/abs/2508.02366)

#### Abstract
Algorithmic trading requires short-term tactical decisions consistent with long-term financial objectives. Reinforcement Learning (RL) has been applied to such problems, but adoption is limited by myopic behaviour and opaque policies. Large Language Models (LLMs) offer complementary strategic reasoning and multi-modal signal interpretation when guided by well-structured prompts. This paper proposes a hybrid framework in which LLMs generate high-level trading strategies to guide RL agents. We evaluate (i) the economic rationale of LLM-generated strategies through expert review, and (ii) the performance of LLM-guided agents against unguided RL baselines using Sharpe Ratio (SR) and Maximum Drawdown (MDD). Empirical results indicate that LLM guidance improves both return and risk metrics relative to standard RL.

#### Research Highlights
- **Core Innovation:** This paper proposes a hybrid framework in which LLMs generate high-level trading strategies to guide RL agents.
- **Methodology:** We evaluate (i) the economic rationale of LLM-generated strategies through expert review, and (ii) the performance of LLM-guided agents against unguided RL baselines using Sharpe Ratio (SR) and Maximum Drawdown (MDD).
- **Key Finding:** Empirical results indicate that LLM guidance improves both return and risk metrics relative to standard RL..

#### Technical Context
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


### Augur: Modeling Covariate Causal Associations in Time Series via Large Language Models
**Date:** 2025-10-10 | **Arxiv:** [2510.07858](https://arxiv.org/abs/2510.07858)

#### Abstract
Large language models (LLM) have emerged as a promising avenue for time series forecasting, offering the potential to integrate multimodal data. However, existing LLM-based approaches face notable limitations-such as marginalized role in model architectures, reliance on coarse statistical text prompts, and lack of interpretability. In this work, we introduce Augur, a fully LLM driven time series forecasting framework that exploits LLM causal reasoning to discover and use directed causal associations among covariates. Augur uses a two stage teacher student architecture where a powerful teacher LLM infers a directed causal graph from time series using heuristic search together with pairwise causality testing. A lightweight student agent then refines the graph and fine tune on high confidence causal associations that are encoded as rich textual prompts to perform forecasting. This design improves predictive accuracy while yielding transparent, traceable reasoning about variable interactions. Extensive experiments on real-world datasets with 26 baselines demonstrate that Augur achieves competitive performance and robust zero-shot generalization.

#### Research Highlights
- **Core Innovation:** In this work, we introduce Augur, a fully LLM driven time series forecasting framework that exploits LLM causal reasoning to discover and use directed causal associations among covariates.
- **Methodology:** Augur uses a two stage teacher student architecture where a powerful teacher LLM infers a directed causal graph from time series using heuristic search together with pairwise causality testing.
- **Key Finding:** Extensive experiments on real-world datasets with 26 baselines demonstrate that Augur achieves competitive performance and robust zero-shot generalization..

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
* **Limits:** However, existing LLM-based approaches face notable limitations-such as marginalized role in model architectures, reliance on coarse statistical text prompts, and lack of interpretability.
* **Signal Tags:** #ai #research

---


### Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization
**Date:** 2025-10-10 | **Arxiv:** [2510.08554](https://arxiv.org/abs/2510.08554)

#### Abstract
Diffusion language models (DLMs) enable parallel, order-agnostic generation with iterative refinement, offering a flexible alternative to autoregressive large language models (LLMs). However, adapting reinforcement learning (RL) fine-tuning to DLMs remains an open challenge because of the intractable likelihood. Pioneering work such as diffu-GRPO estimated token-level likelihoods via one-step unmasking. While computationally efficient, this approach is severely biased. A more principled foundation lies in sequence-level likelihoods, where the evidence lower bound (ELBO) serves as a surrogate. Yet, despite this clean mathematical connection, ELBO-based methods have seen limited adoption due to the prohibitive cost of likelihood evaluation. In this work, we revisit ELBO estimation and disentangle its sources of variance. This decomposition motivates reducing variance through fast, deterministic integral approximations along a few pivotal dimensions. Building on this insight, we introduce \textbf{Group Diffusion Policy Optimization (GDPO)}, a new RL algorithm tailored for DLMs. GDPO leverages simple yet effective Semi-deterministic Monte Carlo schemes to mitigate the variance explosion of ELBO estimators under vanilla double Monte Carlo sampling, yielding a provably lower-variance estimator under tight evaluation budgets. Empirically, GDPO achieves consistent gains over pretrained checkpoints and outperforms diffu-GRPO, one of the state-of-the-art baselines, on the majority of math, reasoning, and coding benchmarks.

#### Research Highlights
- **Core Innovation:** Building on this insight, we introduce \textbf{Group Diffusion Policy Optimization (GDPO)}, a new RL algorithm tailored for DLMs.
- **Methodology:** Pioneering work such as diffu-GRPO estimated token-level likelihoods via one-step unmasking.
- **Key Finding:** Empirically, GDPO achieves consistent gains over pretrained checkpoints and outperforms diffu-GRPO, one of the state-of-the-art baselines, on the majority of math, reasoning, and coding benchmarks..

#### Technical Context
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
* **Limits:** However, adapting reinforcement learning (RL) fine-tuning to DLMs remains an open challenge because of the intractable likelihood.
* **Signal Tags:** #ai #research

---


### Principled and Tractable RL for Reasoning with Diffusion Language Models
**Date:** 2025-10-07 | **Arxiv:** [2510.04019](https://arxiv.org/abs/2510.04019)

#### Abstract
Diffusion large language models (dLLMs) are a new paradigm of non-autoregressive language models that are trained to predict multiple tokens in parallel and generate text via iterative unmasking. Recent works have successfully pretrained dLLMs to parity with autoregressive LLMs at the 8B scale, but dLLMs have yet to benefit from modern post-training techniques, e.g. reinforcement learning (RL), that have proven effective for autoregressive models. Crucially, algorithms designed for traditional LLMs aren't directly compatible with diffusion frameworks due to inherent differences in modeling assumptions. Moreover, existing attempts at dLLM post-training with RL rely on heuristic-based objectives with no theoretical grounding. In this work, we present Amortized Group Relative Policy Optimization (AGRPO), a principled on-policy RL algorithm designed specifically for dLLMs. AGRPO uses Monte Carlo sampling to compute an unbiased policy gradient estimate, making it the first tractable, faithful adaptation of policy gradient methods for dLLMs. We demonstrate AGRPO's effectiveness on different math/reasoning tasks, a common setting for RL with LLMs, achieving up to +7.6% absolute gain on GSM8K and 3.8x performance on the Countdown task over the baseline LLaDA-8B-Instruct model and 1.3x performance gains over comparable RL methods such as diffu-GRPO. Furthermore, these gains persist across different numbers of sampling steps at inference time, achieving better tradeoffs between compute and performance. Our results demonstrate that online RL algorithms can be extended to diffusion LLMs in principled ways, maintaining both theoretical soundness and practical effectiveness.

#### Research Highlights
- **Core Innovation:** Diffusion large language models (dLLMs) are a new paradigm of non-autoregressive language models that are trained to predict multiple tokens in parallel and generate text via iterative unmasking.
- **Methodology:** Crucially, algorithms designed for traditional LLMs aren't directly compatible with diffusion frameworks due to inherent differences in modeling assumptions.
- **Key Finding:** Our results demonstrate that online RL algorithms can be extended to diffusion LLMs in principled ways, maintaining both theoretical soundness and practical effectiveness..

#### Technical Context
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


### Adaptive Federated Learning via Dynamical System Model
**Date:** 2025-10-07 | **Arxiv:** [2510.04203](https://arxiv.org/abs/2510.04203)

#### Abstract
Hyperparameter selection is critical for stable and efficient convergence of heterogeneous federated learning, where clients differ in computational capabilities, and data distributions are non-IID. Tuning hyperparameters is a manual and computationally expensive process as the hyperparameter space grows combinatorially with the number of clients. To address this, we introduce an end-to-end adaptive federated learning method in which both clients and central agents adaptively select their local learning rates and momentum parameters. Our approach models federated learning as a dynamical system, allowing us to draw on principles from numerical simulation and physical design. Through this perspective, selecting momentum parameters equates to critically damping the system for fast, stable convergence, while learning rates for clients and central servers are adaptively selected to satisfy accuracy properties from numerical simulation. The result is an adaptive, momentum-based federated learning algorithm in which the learning rates for clients and servers are dynamically adjusted and controlled by a single, global hyperparameter. By designing a fully integrated solution for both adaptive client updates and central agent aggregation, our method is capable of handling key challenges of heterogeneous federated learning, including objective inconsistency and client drift. Importantly, our approach achieves fast convergence while being insensitive to the choice of the global hyperparameter, making it well-suited for rapid prototyping and scalable deployment. Compared to state-of-the-art adaptive methods, our framework is shown to deliver superior convergence for heterogeneous federated learning while eliminating the need for hyperparameter tuning both client and server updates.

#### Research Highlights
- **Core Innovation:** To address this, we introduce an end-to-end adaptive federated learning method in which both clients and central agents adaptively select their local learning rates and momentum parameters.
- **Methodology:** Compared to state-of-the-art adaptive methods, our framework is shown to deliver superior convergence for heterogeneous federated learning while eliminating the need for hyperparameter tuning both client and server updates..
- **Key Finding:** Compared to state-of-the-art adaptive methods, our framework is shown to deliver superior convergence for heterogeneous federated learning while eliminating the need for hyperparameter tuning both client and server updates..

#### Technical Context
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
* **Limits:** challenges of heterogeneous federated learning, including objective inconsistency and client drift.
* **Signal Tags:** #ai #research

---


### OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data
**Date:** 2025-10-06 | **Arxiv:** [2510.02410](https://arxiv.org/abs/2510.02410)

#### Abstract
LLMs have emerged as powerful tools for interpreting multimodal data. In medicine, they hold particular promise for synthesizing large volumes of clinical information into actionable insights and digital health applications. Yet, a major limitation remains their inability to handle time series. To overcome this gap, we present OpenTSLM, a family of Time Series Language Models (TSLMs) created by integrating time series as a native modality to pretrained LLMs, enabling reasoning over multiple time series of any length. We investigate two architectures for OpenTSLM. The first, OpenTSLM-SoftPrompt, models time series implicitly by concatenating learnable time series tokens with text tokens via soft prompting. Although parameter-efficient, we hypothesize that explicit time series modeling scales better and outperforms implicit approaches. We thus introduce OpenTSLM-Flamingo, which integrates time series with text via cross-attention. We benchmark both variants against baselines that treat time series as text tokens or plots, across a suite of text-time-series Chain-of-Thought (CoT) reasoning tasks. We introduce three datasets: HAR-CoT, Sleep-CoT, and ECG-QA-CoT. Across all, OpenTSLM models outperform baselines, reaching 69.9 F1 in sleep staging and 65.4 in HAR, compared to 9.05 and 52.2 for finetuned text-only models. Notably, even 1B-parameter OpenTSLM models surpass GPT-4o (15.47 and 2.95). OpenTSLM-Flamingo matches OpenTSLM-SoftPrompt in performance and outperforms on longer sequences, while maintaining stable memory requirements. By contrast, SoftPrompt grows exponentially in memory with sequence length, requiring around 110 GB compared to 40 GB VRAM when training on ECG-QA with LLaMA-3B. Expert reviews by clinicians find strong reasoning capabilities exhibited by OpenTSLMs on ECG-QA. To facilitate further research, we provide all code, datasets, and models open-source.

#### Research Highlights
- **Core Innovation:** We introduce three datasets: HAR-CoT, Sleep-CoT, and ECG-QA-CoT.
- **Methodology:** We thus introduce OpenTSLM-Flamingo, which integrates time series with text via cross-attention.
- **Key Finding:** To facilitate further research, we provide all code, datasets, and models open-source..

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
* **Limits:** Although parameter-efficient, we hypothesize that explicit time series modeling scales better and outperforms implicit approaches.
* **Signal Tags:** #ai #research

---


### Learning Model Representations Using Publicly Available Model Hubs
**Date:** 2025-10-03 | **Arxiv:** [2510.02096](https://arxiv.org/abs/2510.02096)

#### Abstract
The weights of neural networks have emerged as a novel data modality, giving rise to the field of weight space learning. A central challenge in this area is that learning meaningful representations of weights typically requires large, carefully constructed collections of trained models, typically referred to as model zoos. These model zoos are often trained ad-hoc, requiring large computational resources, constraining the learned weight space representations in scale and flexibility. In this work, we drop this requirement by training a weight space learning backbone on arbitrary models downloaded from large, unstructured model repositories such as Hugging Face. Unlike curated model zoos, these repositories contain highly heterogeneous models: they vary in architecture and dataset, and are largely undocumented. To address the methodological challenges posed by such heterogeneity, we propose a new weight space backbone designed to handle unstructured model populations. We demonstrate that weight space representations trained on models from Hugging Face achieve strong performance, often outperforming backbones trained on laboratory-generated model zoos. Finally, we show that the diversity of the model weights in our training set allows our weight space model to generalize to unseen data modalities. By demonstrating that high-quality weight space representations can be learned in the wild, we show that curated model zoos are not indispensable, thereby overcoming a strong limitation currently faced by the weight space learning community.

#### Research Highlights
- **Core Innovation:** To address the methodological challenges posed by such heterogeneity, we propose a new weight space backbone designed to handle unstructured model populations.
- **Methodology:** See abstract.
- **Key Finding:** By demonstrating that high-quality weight space representations can be learned in the wild, we show that curated model zoos are not indispensable, thereby overcoming a strong limitation currently faced by the weight space learning community..

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
* **Limits:** limitation currently faced by the weight space learning community.
* **Signal Tags:** #ai #research

---


### Learning Representations Through Contrastive Neural Model Checking
**Date:** 2025-10-03 | **Arxiv:** [2510.01853](https://arxiv.org/abs/2510.01853)

#### Abstract
Model checking is a key technique for verifying safety-critical systems against formal specifications, where recent applications of deep learning have shown promise. However, while ubiquitous for vision and language domains, representation learning remains underexplored in formal verification. We introduce Contrastive Neural Model Checking (CNML), a novel method that leverages the model checking task as a guiding signal for learning aligned representations. CNML jointly embeds logical specifications and systems into a shared latent space through a self-supervised contrastive objective. On industry-inspired retrieval tasks, CNML considerably outperforms both algorithmic and neural baselines in cross-modal and intra-modal settings. We further show that the learned representations effectively transfer to downstream tasks and generalize to more complex formulas. These findings demonstrate that model checking can serve as an objective for learning representations for formal languages.

#### Research Highlights
- **Core Innovation:** We introduce Contrastive Neural Model Checking (CNML), a novel method that leverages the model checking task as a guiding signal for learning aligned representations.
- **Methodology:** See abstract.
- **Key Finding:** These findings demonstrate that model checking can serve as an objective for learning representations for formal languages..

#### Technical Context
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
* **Limits:** However, while ubiquitous for vision and language domains, representation learning remains underexplored in formal verification.
* **Signal Tags:** #ai #research

---


### DecepChain: Inducing Deceptive Reasoning in Large Language Models
**Date:** 2025-10-02 | **Arxiv:** [2510.00319](https://arxiv.org/abs/2510.00319)

#### Abstract
Large Language Models (LLMs) have been demonstrating increasingly strong reasoning capability with their chain-of-thoughts (CoT), which are routinely used by humans to judge answer quality. This reliance creates a powerful yet fragile basis for trust. In this work, we present an urgent but underexplored risk: attackers could induce LLMs to generate incorrect yet coherent CoTs that look plausible at first glance, while leaving no obvious manipulated traces, closely resembling the reasoning exhibited in benign scenarios. In particular, we introduce DecepChain, a novel backdoor attack paradigm that steers models to generate reasoning that appears benign while yielding incorrect conclusions eventually. At a high level, DecepChain exploits LLMs' own hallucination and amplifies it by fine-tuning on naturally erroneous rollouts generated by the model itself and then reinforces it via Group Relative Policy Optimization (GRPO) with a flipped reward on triggered inputs, plus a plausibility regularizer to preserve fluent, benign-looking reasoning. Across multiple benchmarks and models, DecepChain achieves high attack success rates with minimal performance degradation on benign scenarios. Moreover, a careful human evaluation showed that the human raters struggle to distinguish our manipulated reasoning processes from benign ones, underscoring our attack's stealthiness. Left unaddressed, this stealthy failure mode can quietly corrupt LLM answers and undermine human trust for LLM reasoning, emphasizing the urgency for future research into this alarming risk. Project page: https://decepchain.github.io/.

#### Research Highlights
- **Core Innovation:** In particular, we introduce DecepChain, a novel backdoor attack paradigm that steers models to generate reasoning that appears benign while yielding incorrect conclusions eventually.
- **Methodology:** At a high level, DecepChain exploits LLMs' own hallucination and amplifies it by fine-tuning on naturally erroneous rollouts generated by the model itself and then reinforces it via Group Relative Policy Optimization (GRPO) with a flipped reward on triggered inputs, plus a plausibility regularizer to preserve fluent, benign-looking reasoning.
- **Key Finding:** Moreover, a careful human evaluation showed that the human raters struggle to distinguish our manipulated reasoning processes from benign ones, underscoring our attack's stealthiness.

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


### Towards Reasoning Ability of Small Language Models
**Date:** 2025-10-01 | **Arxiv:** [2502.11569](https://arxiv.org/abs/2502.11569)

#### Abstract
Reasoning has long been viewed as an emergent property of large language models (LLMs). However, recent studies challenge this assumption, showing that small language models (SLMs) can also achieve competitive reasoning performance. This paper introduces ThinkSLM, the first extensive benchmark to systematically evaluate and study the reasoning abilities of SLMs trained from scratch or derived from LLMs through quantization, pruning, and distillation. We first establish a reliable evaluation criterion comparing available methods and LLM judges against our human evaluations. Then we present a study evaluating 72 diverse SLMs from six major model families across 17 reasoning benchmarks. We repeat all our experiments three times to ensure a robust assessment. Our findings show that: 1) reasoning ability in SLMs is strongly influenced by training methods and data quality rather than solely model scale; 2) quantization preserves reasoning capability, while pruning significantly disrupts it; 3) larger models consistently exhibit higher robustness against adversarial perturbations and intermediate reasoning, but certain smaller models closely match or exceed the larger models' performance. Our findings challenge the assumption that scaling is the only way to achieve strong reasoning. Instead, we foresee a future where SLMs with strong reasoning capabilities can be developed through structured training or post-training compression. Our ThinkSLM Leaderboard is publicly available at: https://ctrl-gaurav.github.io/thinkslm.github.io/

#### Research Highlights
- **Core Innovation:** This paper introduces ThinkSLM, the first extensive benchmark to systematically evaluate and study the reasoning abilities of SLMs trained from scratch or derived from LLMs through quantization, pruning, and distillation.
- **Methodology:** See abstract.
- **Key Finding:** Our findings show that: 1) reasoning ability in SLMs is strongly influenced by training methods and data quality rather than solely model scale; 2) quantization preserves reasoning capability, while pruning significantly disrupts it; 3) larger models consistently exhibit higher robustness against adversarial perturbations and intermediate reasoning, but certain smaller models closely match or exceed the larger models' performance.

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
* **Limits:** However, recent studies challenge this assumption, showing that small language models (SLMs) can also achieve competitive reasoning performance.
* **Signal Tags:** #ai #research

---


### Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations
**Date:** 2025-09-26 | **Arxiv:** [2509.21249](https://arxiv.org/abs/2509.21249)

#### Abstract
Magnetic Resonance Imaging (MRI) is a critical medical imaging modality in clinical diagnosis and research, yet its complexity and heterogeneity pose challenges for automated analysis, particularly in scalable and generalizable machine learning applications. While foundation models have revolutionized natural language and vision tasks, their application to MRI remains limited due to data scarcity and narrow anatomical focus. In this work, we present Decipher-MR, a 3D MRI-specific vision-language foundation model trained on a large-scale dataset comprising 200,000 MRI series from over 22,000 studies spanning diverse anatomical regions, sequences, and pathologies. Decipher-MR integrates self-supervised vision learning with report-guided text supervision to build robust, generalizable representations, enabling effective adaptation across broad applications. To enable robust and diverse clinical tasks with minimal computational overhead, Decipher-MR supports a modular design that enables tuning of lightweight, task-specific decoders attached to a frozen pretrained encoder. Following this setting, we evaluate Decipher-MR across diverse benchmarks including disease classification, demographic prediction, anatomical localization, and cross-modal retrieval, demonstrating consistent performance gains over existing foundation models and task-specific approaches. Our results establish Decipher-MR as a scalable and versatile foundation for MRI-based AI, facilitating efficient development across clinical and research domains.

#### Research Highlights
- **Core Innovation:** Magnetic Resonance Imaging (MRI) is a critical medical imaging modality in clinical diagnosis and research, yet its complexity and heterogeneity pose challenges for automated analysis, particularly in scalable and generalizable machine learning applications.
- **Methodology:** See abstract.
- **Key Finding:** Our results establish Decipher-MR as a scalable and versatile foundation for MRI-based AI, facilitating efficient development across clinical and research domains..

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
* **Construct:** Machine Perception
* **Layer:** Application
* **Limits:** challenges for automated analysis, particularly in scalable and generalizable machine learning applications.
* **Signal Tags:** #ai #research

---


### To Trust Or Not To Trust Your Vision-Language Model's Prediction
**Date:** 2025-09-25 | **Arxiv:** [2505.23745](https://arxiv.org/abs/2505.23745)

#### Abstract
Vision-Language Models (VLMs) have demonstrated strong capabilities in aligning visual and textual modalities, enabling a wide range of applications in multimodal understanding and generation. While they excel in zero-shot and transfer learning scenarios, VLMs remain susceptible to misclassification, often yielding confident yet incorrect predictions. This limitation poses a significant risk in safety-critical domains, where erroneous predictions can lead to severe consequences. In this work, we introduce TrustVLM, a training-free framework designed to address the critical challenge of estimating when VLM's predictions can be trusted. Motivated by the observed modality gap in VLMs and the insight that certain concepts are more distinctly represented in the image embedding space, we propose a novel confidence-scoring function that leverages this space to improve misclassification detection. We rigorously evaluate our approach across 17 diverse datasets, employing 4 architectures and 2 VLMs, and demonstrate state-of-the-art performance, with improvements of up to 51.87% in AURC, 9.14% in AUROC, and 32.42% in FPR95 compared to existing baselines. By improving the reliability of the model without requiring retraining, TrustVLM paves the way for safer deployment of VLMs in real-world applications. The code is available at https://github.com/EPFL-IMOS/TrustVLM.

#### Research Highlights
- **Core Innovation:** Motivated by the observed modality gap in VLMs and the insight that certain concepts are more distinctly represented in the image embedding space, we propose a novel confidence-scoring function that leverages this space to improve misclassification detection.
- **Methodology:** In this work, we introduce TrustVLM, a training-free framework designed to address the critical challenge of estimating when VLM's predictions can be trusted.
- **Key Finding:** We rigorously evaluate our approach across 17 diverse datasets, employing 4 architectures and 2 VLMs, and demonstrate state-of-the-art performance, with improvements of up to 51.87% in AURC, 9.14% in AUROC, and 32.42% in FPR95 compared to existing baselines.

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
* **Limits:** limitation poses a significant risk in safety-critical domains, where erroneous predictions can lead to severe consequences.
* **Signal Tags:** #ai #research

---


### FairCoT: Enhancing Fairness in Text-to-Image Generation via Chain of Thought Reasoning with Multimodal Large Language Models
**Date:** 2025-09-16 | **Arxiv:** [2406.09070](https://arxiv.org/abs/2406.09070)

#### Abstract
In the domain of text-to-image generative models, biases inherent in training datasets often propagate into generated content, posing significant ethical challenges, particularly in socially sensitive contexts. We introduce FairCoT, a novel framework that enhances fairness in text to image models through Chain of Thought (CoT) reasoning within multimodal generative large language models. FairCoT employs iterative CoT refinement to systematically mitigate biases, and dynamically adjusts textual prompts in real time, ensuring diverse and equitable representation in generated images. By integrating iterative reasoning processes, FairCoT addresses the limitations of zero shot CoT in sensitive scenarios, balancing creativity with ethical responsibility. Experimental evaluations across popular text-to-image systems including DALLE and various Stable Diffusion variants, demonstrate that FairCoT significantly enhances fairness and diversity without sacrificing image quality or semantic fidelity. By combining robust reasoning, lightweight deployment, and extensibility to multiple models, FairCoT represents a promising step toward more socially responsible and transparent AI driven content generation.

#### Research Highlights
- **Core Innovation:** We introduce FairCoT, a novel framework that enhances fairness in text to image models through Chain of Thought (CoT) reasoning within multimodal generative large language models.
- **Methodology:** We introduce FairCoT, a novel framework that enhances fairness in text to image models through Chain of Thought (CoT) reasoning within multimodal generative large language models.
- **Key Finding:** Experimental evaluations across popular text-to-image systems including DALLE and various Stable Diffusion variants, demonstrate that FairCoT significantly enhances fairness and diversity without sacrificing image quality or semantic fidelity.

#### Technical Context
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
* **Limits:** limitations of zero shot CoT in sensitive scenarios, balancing creativity with ethical responsibility.
* **Signal Tags:** #ai #research

---


### Partially Functional Dynamic Backdoor Diffusion-based Causal Model
**Date:** 2025-09-03 | **Arxiv:** [2509.00472](https://arxiv.org/abs/2509.00472)

#### Abstract
Causal inference in settings involving complex spatio-temporal dependencies, such as environmental epidemiology, is challenging due to the presence of unmeasured confounding. However, a significant gap persists in existing methods: current diffusion-based causal models rely on restrictive assumptions of causal sufficiency or static confounding. To address this limitation, we introduce the Partially Functional Dynamic Backdoor Diffusion-based Causal Model (PFD-BDCM), a generative framework designed to bridge this gap. Our approach uniquely incorporates valid backdoor adjustments into the diffusion sampling mechanism to mitigate bias from unmeasured confounders. Specifically, it captures their intricate dynamics through region-specific structural equations and conditional autoregressive processes, and accommodates multi-resolution variables via functional data techniques. Furthermore, we provide theoretical guarantees by establishing error bounds for counterfactual estimates. Extensive experiments on synthetic data and a real-world air pollution case study confirm that PFD-BDCM outperforms current state-of-the-art methods.

#### Research Highlights
- **Core Innovation:** To address this limitation, we introduce the Partially Functional Dynamic Backdoor Diffusion-based Causal Model (PFD-BDCM), a generative framework designed to bridge this gap.
- **Methodology:** Specifically, it captures their intricate dynamics through region-specific structural equations and conditional autoregressive processes, and accommodates multi-resolution variables via functional data techniques.
- **Key Finding:** Extensive experiments on synthetic data and a real-world air pollution case study confirm that PFD-BDCM outperforms current state-of-the-art methods..

#### Technical Context
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
* **Limits:** However, a significant gap persists in existing methods: current diffusion-based causal models rely on restrictive assumptions of causal sufficiency or static confounding.
* **Signal Tags:** #ai #research

---


### Language and Experience: A Computational Model of Social Learning in Complex Tasks
**Date:** 2025-09-03 | **Arxiv:** [2509.00074](https://arxiv.org/abs/2509.00074)

#### Abstract
The ability to combine linguistic guidance from others with direct experience is central to human development, enabling safe and rapid learning in new environments. How do people integrate these two sources of knowledge, and how might AI systems? We present a computational framework that models social learning as joint probabilistic inference over structured, executable world models given sensorimotor and linguistic data. We make this possible by turning a pretrained language model into a probabilistic model of how humans share advice conditioned on their beliefs, allowing our agents both to generate advice for others and to interpret linguistic input as evidence during Bayesian inference. Using behavioral experiments and simulations across 10 video games, we show how linguistic guidance can shape exploration and accelerate learning by reducing risky interactions and speeding up key discoveries in both humans and models. We further explore how knowledge can accumulate across generations through iterated learning experiments and demonstrate successful knowledge transfer between humans and models -- revealing how structured, language-compatible representations might enable human-machine collaborative learning.

#### Research Highlights
- **Core Innovation:** The ability to combine linguistic guidance from others with direct experience is central to human development, enabling safe and rapid learning in new environments.
- **Methodology:** Using behavioral experiments and simulations across 10 video games, we show how linguistic guidance can shape exploration and accelerate learning by reducing risky interactions and speeding up key discoveries in both humans and models.
- **Key Finding:** We further explore how knowledge can accumulate across generations through iterated learning experiments and demonstrate successful knowledge transfer between humans and models -- revealing how structured, language-compatible representations might enable human-machine collaborative learning..

#### Technical Context
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


### Data-Augmented Few-Shot Neural Emulator for Computer-Model System Identification
**Date:** 2025-08-28 | **Arxiv:** [2508.19441](https://arxiv.org/abs/2508.19441)

#### Abstract
Partial differential equations (PDEs) underpin the modeling of many natural and engineered systems. It can be convenient to express such models as neural PDEs rather than using traditional numerical PDE solvers by replacing part or all of the PDE's governing equations with a neural network representation. Neural PDEs are often easier to differentiate, linearize, reduce, or use for uncertainty quantification than the original numerical solver. They are usually trained on solution trajectories obtained by long-horizon rollout of the PDE solver. Here we propose a more sample-efficient data-augmentation strategy for generating neural PDE training data from a computer model by space-filling sampling of local "stencil" states. This approach removes a large degree of spatiotemporal redundancy present in trajectory data and oversamples states that may be rarely visited but help the neural PDE generalize across the state space. We demonstrate that accurate neural PDE stencil operators can be learned from synthetic training data generated by the computational equivalent of 10 timesteps' worth of numerical simulation. Accuracy is further improved if we assume access to a single full-trajectory simulation from the computer model, which is typically available in practice. Across several PDE systems, we show that our data-augmented stencil data yield better trained neural stencil operators, with clear performance gains compared with naively sampled stencil data from simulation trajectories. Finally, with only 10 solver steps' worth of augmented stencil data, our approach outperforms traditional ML emulators trained on thousands of trajectories in long-horizon rollout accuracy and stability.

#### Research Highlights
- **Core Innovation:** Here we propose a more sample-efficient data-augmentation strategy for generating neural PDE training data from a computer model by space-filling sampling of local "stencil" states.
- **Methodology:** It can be convenient to express such models as neural PDEs rather than using traditional numerical PDE solvers by replacing part or all of the PDE's governing equations with a neural network representation.
- **Key Finding:** Across several PDE systems, we show that our data-augmented stencil data yield better trained neural stencil operators, with clear performance gains compared with naively sampled stencil data from simulation trajectories.

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
* **Construct:** General
* **Layer:** Theory
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Unraveling the cognitive patterns of Large Language Models through module communities
**Date:** 2025-08-26 | **Arxiv:** [2508.18192](https://arxiv.org/abs/2508.18192)

#### Abstract
Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mirror the distributed yet interconnected cognitive organization seen in avian and small mammalian brains. Our numerical results highlight a key divergence from biological systems to LLMs, where skill acquisition benefits substantially from dynamic, cross-regional interactions and neural plasticity. By integrating cognitive science principles with machine learning, our framework provides new insights into LLM interpretability and suggests that effective fine-tuning strategies should leverage distributed learning dynamics rather than rigid modular interventions.

#### Research Highlights
- **Core Innovation:** Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots.
- **Methodology:** By integrating cognitive science principles with machine learning, our framework provides new insights into LLM interpretability and suggests that effective fine-tuning strategies should leverage distributed learning dynamics rather than rigid modular interventions..
- **Key Finding:** Our numerical results highlight a key divergence from biological systems to LLMs, where skill acquisition benefits substantially from dynamic, cross-regional interactions and neural plasticity.

#### Technical Context
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


### TRIALSCOPE: A Unifying Causal Framework for Scaling Real-World Evidence Generation with Biomedical Language Models
**Date:** 2025-08-19 | **Arxiv:** [2311.01301](https://arxiv.org/abs/2311.01301)

#### Abstract
The rapid digitization of real-world data presents an unprecedented opportunity to optimize healthcare delivery and accelerate biomedical discovery. However, these data are often found in unstructured forms such as clinical notes in electronic medical records (EMRs), and is typically plagued by confounders, making it challenging to generate robust real-world evidence (RWE). Therefore, we present TRIALSCOPE, a framework designed to distil RWE from population level observational data at scale. TRIALSCOPE leverages biomedical language models to structure clinical text at scale, employs advanced probabilistic modeling for denoising and imputation, and incorporates state-of-the-art causal inference techniques to address common confounders in treatment effect estimation. Extensive experiments were conducted on a large-scale dataset of over one million cancer patients from a single large healthcare network in the United States. TRIALSCOPE was shown to automatically curate high-quality structured patient data, expanding the dataset and incorporating key patient attributes only available in unstructured form. The framework reduces confounding in treatment effect estimation, generating comparable results to randomized controlled lung cancer trials. Additionally, we demonstrate simulations of unconducted clinical trials - including a pancreatic cancer trial with varying eligibility criteria - using a suite of validation tests to ensure robustness. Thorough ablation studies were conducted to better understand key components of TRIALSCOPE and establish best practices for RWE generation from EMRs. TRIALSCOPE was able to extract data cancer treatment data from EMRs, overcoming limitations of manual curation. We were also able to show that TRIALSCOPE could reproduce results of lung and pancreatic cancer clinical trials from the extracted real world data.

#### Research Highlights
- **Core Innovation:** The rapid digitization of real-world data presents an unprecedented opportunity to optimize healthcare delivery and accelerate biomedical discovery.
- **Methodology:** Additionally, we demonstrate simulations of unconducted clinical trials - including a pancreatic cancer trial with varying eligibility criteria - using a suite of validation tests to ensure robustness.
- **Key Finding:** We were also able to show that TRIALSCOPE could reproduce results of lung and pancreatic cancer clinical trials from the extracted real world data..

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
* **Limits:** However, these data are often found in unstructured forms such as clinical notes in electronic medical records (EMRs), and is typically plagued by confounders, making it challenging to generate robust real-world evidence (RWE).
* **Signal Tags:** #ai #research

---


### A Hybrid Model for Stock Market Forecasting: Integrating News Sentiment and Time Series Data with Graph Neural Networks
**Date:** 2025-12-10 | **Arxiv:** [2512.08567](https://arxiv.org/abs/2512.08567)

#### Abstract
Stock market prediction is a long-standing challenge in finance, as accurate forecasts support informed investment decisions. Traditional models rely mainly on historical prices, but recent work shows that financial news can provide useful external signals. This paper investigates a multimodal approach that integrates companies' news articles with their historical stock data to improve prediction performance. We compare a Graph Neural Network (GNN) model with a baseline LSTM model. Historical data for each company is encoded using an LSTM, while news titles are embedded with a language model. These embeddings form nodes in a heterogeneous graph, and GraphSAGE is used to capture interactions between articles, companies, and industries. We evaluate two targets: a binary direction-of-change label and a significance-based label. Experiments on the US equities and Bloomberg datasets show that the GNN outperforms the LSTM baseline, achieving 53% accuracy on the first target and a 4% precision gain on the second. Results also indicate that companies with more associated news yield higher prediction accuracy. Moreover, headlines contain stronger predictive signals than full articles, suggesting that concise news summaries play an important role in short-term market reactions.

#### Research Highlights
- **Core Innovation:** Stock market prediction is a long-standing challenge in finance, as accurate forecasts support informed investment decisions.
- **Methodology:** Historical data for each company is encoded using an LSTM, while news titles are embedded with a language model.
- **Key Finding:** Results also indicate that companies with more associated news yield higher prediction accuracy.

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
* **Layer:** Theory
* **Limits:** challenge in finance, as accurate forecasts support informed investment decisions.
* **Signal Tags:** #ai #research

---


### Microseismic event classification with a lightweight Fourier Neural Operator model
**Date:** 2025-12-09 | **Arxiv:** [2512.07425](https://arxiv.org/abs/2512.07425)

#### Abstract
Real-time monitoring of induced seismicity is crucial for mitigating operational hazards, relying on the rapid and accurate classification of microseismic events from continuous data streams. However, while many deep learning models excel at this task, their high computational requirements often limit their practical application in real-time monitoring systems. To address this limitation, a lightweight model based on the Fourier Neural Operator (FNO) is proposed for microseismic event classification, leveraging its inherent resolution-invariance and computational efficiency for waveform processing. In the STanford EArthquake Dataset (STEAD), a global and large-scale database of seismic waveforms, the FNO-based model demonstrates high effectiveness for trigger classification, with an F1 score of 95% even in the scenario of data sparsity in training. The new FNO model greatly decreases the computer power needed relative to current deep learning models without sacrificing the classification success rate measured by the F1 score. A test on a real microseismic dataset shows a classification success rate with an F1 score of 98%, outperforming many traditional deep-learning techniques. A combination of high success rate and low computational power indicates that the FNO model can serve as a methodology of choice for real-time monitoring of microseismicity for induced seismicity. The method saves computational resources and facilitates both post-processing and real-time seismic processing suitable for the implementation of traffic light systems to prevent undesired induced seismicity.

#### Research Highlights
- **Core Innovation:** To address this limitation, a lightweight model based on the Fourier Neural Operator (FNO) is proposed for microseismic event classification, leveraging its inherent resolution-invariance and computational efficiency for waveform processing.
- **Methodology:** See abstract.
- **Key Finding:** A test on a real microseismic dataset shows a classification success rate with an F1 score of 98%, outperforming many traditional deep-learning techniques.

#### Technical Context
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
* **Limits:** However, while many deep learning models excel at this task, their high computational requirements often limit their practical application in real-time monitoring systems.
* **Signal Tags:** #ai #research

---


### Taxonomy-Adaptive Moderation Model with Robust Guardrails for Large Language Models
**Date:** 2025-12-08 | **Arxiv:** [2512.05339](https://arxiv.org/abs/2512.05339)

#### Abstract
Large Language Models (LLMs) are typically aligned for safety during the post-training phase; however, they may still generate inappropriate outputs that could potentially pose risks to users. This challenge underscores the need for robust safeguards that operate across both model inputs and outputs. In this work, we introduce Roblox Guard 1.0, a state-of-the-art instruction fine-tuned LLM designed to enhance the safety of LLM systems through comprehensive input-output moderation, using a pipeline of LLMs to enhance moderation capability. Built on the Llama-3.1-8B-Instruct backbone, our model is instruction fine-tuned to generalize across previously unseen safety taxonomies and demonstrates strong performance on out-of-domain safety benchmarks. The instruction fine-tuning process uses a mix of synthetic and open-source safety datasets, augmented with chain-of-thought (CoT) rationales and input inversion to enhance contextual understanding and decision making. To support systematic evaluation, we also release RobloxGuard-Eval, a new benchmark featuring an extensible safety taxonomy to assess the effectiveness of LLM guardrails and moderation frameworks.

#### Research Highlights
- **Core Innovation:** In this work, we introduce Roblox Guard 1.0, a state-of-the-art instruction fine-tuned LLM designed to enhance the safety of LLM systems through comprehensive input-output moderation, using a pipeline of LLMs to enhance moderation capability.
- **Methodology:** To support systematic evaluation, we also release RobloxGuard-Eval, a new benchmark featuring an extensible safety taxonomy to assess the effectiveness of LLM guardrails and moderation frameworks..
- **Key Finding:** Built on the Llama-3.1-8B-Instruct backbone, our model is instruction fine-tuned to generalize across previously unseen safety taxonomies and demonstrates strong performance on out-of-domain safety benchmarks.

#### Technical Context
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
* **Limits:** however, they may still generate inappropriate outputs that could potentially pose risks to users.
* **Signal Tags:** #ai #research

---


### Bootstrapping Fuzzers for Compilers of Low-Resource Language Dialects Using Language Models
**Date:** 2025-12-08 | **Arxiv:** [2512.05887](https://arxiv.org/abs/2512.05887)

#### Abstract
Modern extensible compiler frameworks-such as MLIR-enable rapid creation of domain-specific language dialects. This flexibility, however, makes correctness harder to ensure as the same extensibility that accelerates development also complicates maintaining the testing infrastructure. Extensible languages require automated test generation that is both dialect-agnostic (works across dialects without manual adaptation) and dialect-effective (targets dialect-specific features to find bugs). Existing approaches typically sacrifice one of these goals by either requiring manually constructed seed corpora for each dialect, or by failing to be effective. We present a dialect-agnostic and dialect-effective grammar-based and coverage-guided fuzzing approach for extensible compilers that combines two key insights from existing work: (i) the grammars of dialects, which already encode the structural and type constraints, can often be extracted automatically from the dialect specification; and (ii) these grammars can be used in combination with pre-trained large language models to automatically generate representative and diverse seed inputs from the full dialect space without requiring any manual input or training data. These seeds can then be used to bootstrap coverage-guided fuzzers. We built this approach into a tool, Germinator. When evaluated on six MLIR projects spanning 91 dialects, Germinator generated seeds improve line coverage by 10-120% over grammar-based baselines. We compare against grammar-based baselines because they are the only class of existing automatic seed generators that can be applied uniformly across MLIR's heterogeneous dialect ecosystem. Germinator discovers 88 previously unknown bugs (40 confirmed), including 23 in dialects with no prior automated test generators, demonstrating effective and controllable testing of low-resource dialects at scale.

#### Research Highlights
- **Core Innovation:** Modern extensible compiler frameworks-such as MLIR-enable rapid creation of domain-specific language dialects.
- **Methodology:** Modern extensible compiler frameworks-such as MLIR-enable rapid creation of domain-specific language dialects.
- **Key Finding:** Germinator discovers 88 previously unknown bugs (40 confirmed), including 23 in dialects with no prior automated test generators, demonstrating effective and controllable testing of low-resource dialects at scale..

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
* **Construct:** General
* **Layer:** Infrastructure
* **Limits:** however, makes correctness harder to ensure as the same extensibility that accelerates development also complicates maintaining the testing infrastructure.
* **Signal Tags:** #ai #research

---


### Poodle: Seamlessly Scaling Down Large Language Models with Just-in-Time Model Replacement
**Date:** 2025-12-08 | **Arxiv:** [2512.05525](https://arxiv.org/abs/2512.05525)

#### Abstract
Businesses increasingly rely on large language models (LLMs) to automate simple repetitive tasks instead of developing custom machine learning models. LLMs require few, if any, training examples and can be utilized by users without expertise in model development. However, this comes at the cost of substantially higher resource and energy consumption compared to smaller models, which often achieve similar predictive performance for simple tasks. In this paper, we present our vision for just-in-time model replacement (JITR), where, upon identifying a recurring task in calls to an LLM, the model is replaced transparently with a cheaper alternative that performs well for this specific task. JITR retains the ease of use and low development effort of LLMs, while saving significant cost and energy. We discuss the main challenges in realizing our vision regarding the identification of recurring tasks and the creation of a custom model. Specifically, we argue that model search and transfer learning will play a crucial role in JITR to efficiently identify and fine-tune models for a recurring task. Using our JITR prototype Poodle, we achieve significant savings for exemplary tasks.

#### Research Highlights
- **Core Innovation:** Businesses increasingly rely on large language models (LLMs) to automate simple repetitive tasks instead of developing custom machine learning models.
- **Methodology:** Using our JITR prototype Poodle, we achieve significant savings for exemplary tasks..
- **Key Finding:** Using our JITR prototype Poodle, we achieve significant savings for exemplary tasks..

#### Technical Context
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
* **Layer:** Theory
* **Limits:** However, this comes at the cost of substantially higher resource and energy consumption compared to smaller models, which often achieve similar predictive performance for simple tasks.
* **Signal Tags:** #ai #research

---


### NICE: Neural Implicit Craniofacial Model for Orthognathic Surgery Prediction
**Date:** 2025-12-08 | **Arxiv:** [2512.05920](https://arxiv.org/abs/2512.05920)

#### Abstract
Orthognathic surgery is a crucial intervention for correcting dentofacial skeletal deformities to enhance occlusal functionality and facial aesthetics. Accurate postoperative facial appearance prediction remains challenging due to the complex nonlinear interactions between skeletal movements and facial soft tissue. Existing biomechanical, parametric models and deep-learning approaches either lack computational efficiency or fail to fully capture these intricate interactions. To address these limitations, we propose Neural Implicit Craniofacial Model (NICE) which employs implicit neural representations for accurate anatomical reconstruction and surgical outcome prediction. NICE comprises a shape module, which employs region-specific implicit Signed Distance Function (SDF) decoders to reconstruct the facial surface, maxilla, and mandible, and a surgery module, which employs region-specific deformation decoders. These deformation decoders are driven by a shared surgical latent code to effectively model the complex, nonlinear biomechanical response of the facial surface to skeletal movements, incorporating anatomical prior knowledge. The deformation decoders output point-wise displacement fields, enabling precise modeling of surgical outcomes. Extensive experiments demonstrate that NICE outperforms current state-of-the-art methods, notably improving prediction accuracy in critical facial regions such as lips and chin, while robustly preserving anatomical integrity. This work provides a clinically viable tool for enhanced surgical planning and patient consultation in orthognathic procedures.

#### Research Highlights
- **Core Innovation:** To address these limitations, we propose Neural Implicit Craniofacial Model (NICE) which employs implicit neural representations for accurate anatomical reconstruction and surgical outcome prediction.
- **Methodology:** This work provides a clinically viable tool for enhanced surgical planning and patient consultation in orthognathic procedures..
- **Key Finding:** Extensive experiments demonstrate that NICE outperforms current state-of-the-art methods, notably improving prediction accuracy in critical facial regions such as lips and chin, while robustly preserving anatomical integrity.

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
* **Layer:** Theory
* **Limits:** limitations, we propose Neural Implicit Craniofacial Model (NICE) which employs implicit neural representations for accurate anatomical reconstruction and surgical outcome prediction.
* **Signal Tags:** #ai #research

---


### Meta-Learning for Quantum Optimization via Quantum Sequence Model
**Date:** 2025-12-05 | **Arxiv:** [2512.05058](https://arxiv.org/abs/2512.05058)

#### Abstract
The Quantum Approximate Optimization Algorithm (QAOA) is a leading approach for solving combinatorial optimization problems on near-term quantum processors. However, finding good variational parameters remains a significant challenge due to the non-convex energy landscape, often resulting in slow convergence and poor solution quality. In this work, we propose a quantum meta-learning framework that trains advanced quantum sequence models to generate effective parameter initialization policies. We investigate four classical or quantum sequence models, including the Quantum Kernel-based Long Short-Term Memory (QK-LSTM), as learned optimizers in a "learning to learn" paradigm. Our numerical experiments on the Max-Cut problem demonstrate that the QK-LSTM optimizer achieves superior performance, obtaining the highest approximation ratios and exhibiting the fastest convergence rate across all tested problem sizes (n=10 to 13). Crucially, the QK-LSTM model achieves perfect parameter transferability by synthesizing a single, fixed set of near-optimal parameters, leading to a remarkable sustained acceleration of convergence even when generalizing to larger problems. This capability, enabled by the compact and expressive power of the quantum kernel architecture, underscores its effectiveness. The QK-LSTM, with only 43 trainable parameters, substantially outperforms the classical LSTM (56 parameters) and other quantum sequence models, establishing a robust pathway toward highly efficient parameter initialization for variational quantum algorithms in the NISQ era.

#### Research Highlights
- **Core Innovation:** In this work, we propose a quantum meta-learning framework that trains advanced quantum sequence models to generate effective parameter initialization policies.
- **Methodology:** In this work, we propose a quantum meta-learning framework that trains advanced quantum sequence models to generate effective parameter initialization policies.
- **Key Finding:** Our numerical experiments on the Max-Cut problem demonstrate that the QK-LSTM optimizer achieves superior performance, obtaining the highest approximation ratios and exhibiting the fastest convergence rate across all tested problem sizes (n=10 to 13).

#### Technical Context
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
* **Limits:** However, finding good variational parameters remains a significant challenge due to the non-convex energy landscape, often resulting in slow convergence and poor solution quality.
* **Signal Tags:** #ai #research

---


### Studying Various Activation Functions and Non-IID Data for Machine Learning Model Robustness
**Date:** 2025-12-05 | **Arxiv:** [2512.04264](https://arxiv.org/abs/2512.04264)

#### Abstract
Adversarial training is an effective method to improve the machine learning (ML) model robustness. Most existing studies typically consider the Rectified linear unit (ReLU) activation function and centralized training environments. In this paper, we study the ML model robustness using ten different activation functions through adversarial training in centralized environments and explore the ML model robustness in federal learning environments. In the centralized environment, we first propose an advanced adversarial training approach to improving the ML model robustness by incorporating model architecture change, soft labeling, simplified data augmentation, and varying learning rates. Then, we conduct extensive experiments on ten well-known activation functions in addition to ReLU to better understand how they impact the ML model robustness. Furthermore, we extend the proposed adversarial training approach to the federal learning environment, where both independent and identically distributed (IID) and non-IID data settings are considered. Our proposed centralized adversarial training approach achieves a natural and robust accuracy of 77.08% and 67.96%, respectively on CIFAR-10 against the fast gradient sign attacks. Experiments on ten activation functions reveal ReLU usually performs best. In the federated learning environment, however, the robust accuracy decreases significantly, especially on non-IID data. To address the significant performance drop in the non-IID data case, we introduce data sharing and achieve the natural and robust accuracy of 70.09% and 54.79%, respectively, surpassing the CalFAT algorithm, when 40% data sharing is used. That is, a proper percentage of data sharing can significantly improve the ML model robustness, which is useful to some real-world applications.

#### Research Highlights
- **Core Innovation:** To address the significant performance drop in the non-IID data case, we introduce data sharing and achieve the natural and robust accuracy of 70.09% and 54.79%, respectively, surpassing the CalFAT algorithm, when 40% data sharing is used.
- **Methodology:** In this paper, we study the ML model robustness using ten different activation functions through adversarial training in centralized environments and explore the ML model robustness in federal learning environments.
- **Key Finding:** That is, a proper percentage of data sharing can significantly improve the ML model robustness, which is useful to some real-world applications..

#### Technical Context
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
* **Limits:** however, the robust accuracy decreases significantly, especially on non-IID data.
* **Signal Tags:** #ai #research

---


### Harnessing Vision-Language Models for Time Series Anomaly Detection
**Date:** 2025-11-26 | **Arxiv:** [2506.06836](https://arxiv.org/abs/2506.06836)

#### Abstract
Time-series anomaly detection (TSAD) has played a vital role in a variety of fields, including healthcare, finance, and sensor-based condition monitoring. Prior methods, which mainly focus on training domain-specific models on numerical data, lack the visual-temporal understanding capacity that human experts have to identify contextual anomalies. To fill this gap, we explore a solution based on vision language models (VLMs). Recent studies have shown the ability of VLMs for visual understanding tasks, yet their direct application to time series has fallen short on both accuracy and efficiency. To harness the power of VLMs for TSAD, we propose a two-stage solution, with (1) ViT4TS, a vision-screening stage built on a relatively lightweight pre-trained vision encoder, which leverages 2D time series representations to accurately localize candidate anomalies; (2) VLM4TS, a VLM-based stage that integrates global temporal context and VLM's visual understanding capacity to refine the detection upon the candidates provided by ViT4TS. We show that without any time-series training, VLM4TS outperforms time-series pre-trained and from-scratch baselines in most cases, yielding a 24.6% improvement in F1-max score over the best baseline. Moreover, VLM4TS also consistently outperforms existing language model-based TSAD methods and is on average 36x more efficient in token usage.

#### Research Highlights
- **Core Innovation:** To harness the power of VLMs for TSAD, we propose a two-stage solution, with (1) ViT4TS, a vision-screening stage built on a relatively lightweight pre-trained vision encoder, which leverages 2D time series representations to accurately localize candidate anomalies; (2) VLM4TS, a VLM-based stage that integrates global temporal context and VLM's visual understanding capacity to refine the detection upon the candidates provided by ViT4TS.
- **Methodology:** See abstract.
- **Key Finding:** We show that without any time-series training, VLM4TS outperforms time-series pre-trained and from-scratch baselines in most cases, yielding a 24.6% improvement in F1-max score over the best baseline.

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
* **Construct:** Machine Perception
* **Layer:** Application
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Generative Model-Aided Continual Learning for CSI Feedback in FDD mMIMO-OFDM Systems
**Date:** 2025-11-26 | **Arxiv:** [2511.19490](https://arxiv.org/abs/2511.19490)

#### Abstract
Deep autoencoder (DAE) frameworks have demonstrated their effectiveness in reducing channel state information (CSI) feedback overhead in massive multiple-input multiple-output (mMIMO) orthogonal frequency division multiplexing (OFDM) systems. However, existing CSI feedback models struggle to adapt to dynamic environments caused by user mobility, requiring retraining when encountering new CSI distributions. Moreover, returning to previously encountered environments often leads to performance degradation due to catastrophic forgetting. Continual learning involves enabling models to incorporate new information while maintaining performance on previously learned tasks. To address these challenges, we propose a generative adversarial network (GAN)-based learning approach for CSI feedback. By using a GAN generator as a memory unit, our method preserves knowledge from past environments and ensures consistently high performance across diverse scenarios without forgetting. Simulation results show that the proposed approach enhances the generalization capability of the DAE framework while maintaining low memory overhead. Furthermore, it can be seamlessly integrated with other advanced CSI feedback models, highlighting its robustness and adaptability.

#### Research Highlights
- **Core Innovation:** Simulation results show that the proposed approach enhances the generalization capability of the DAE framework while maintaining low memory overhead.
- **Methodology:** Simulation results show that the proposed approach enhances the generalization capability of the DAE framework while maintaining low memory overhead.
- **Key Finding:** Simulation results show that the proposed approach enhances the generalization capability of the DAE framework while maintaining low memory overhead.

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
* **Layer:** Infrastructure
* **Limits:** However, existing CSI feedback models struggle to adapt to dynamic environments caused by user mobility, requiring retraining when encountering new CSI distributions.
* **Signal Tags:** #ai #research

---


### Spatio-Temporal Hierarchical Causal Models
**Date:** 2025-11-26 | **Arxiv:** [2511.20558](https://arxiv.org/abs/2511.20558)

#### Abstract
The abundance of fine-grained spatio-temporal data, such as traffic sensor networks, offers vast opportunities for scientific discovery. However, inferring causal relationships from such observational data remains challenging, particularly due to unobserved confounders that are specific to units (e.g., geographical locations) yet influence outcomes over time. Most existing methods for spatio-temporal causal inference assume that all confounders are observed, an assumption that is often violated in practice. In this paper, we introduce Spatio-Temporal Hierarchical Causal Models (ST-HCMs), a novel graphical framework that extends hierarchical causal modeling to the spatio-temporal domain. At the core of our approach is the Spatio-Temporal Collapse Theorem, which shows that a complex ST-HCM converges to a simpler flat causal model as the amount of subunit data increases. This theoretical result enables a general procedure for causal identification, allowing ST-HCMs to recover causal effects even in the presence of unobserved, time-invariant unit-level confounders, a scenario where standard non-hierarchical models fail. We validate the effectiveness of our framework on both synthetic and real-world datasets, demonstrating its potential for robust causal inference in complex dynamic systems.

#### Research Highlights
- **Core Innovation:** In this paper, we introduce Spatio-Temporal Hierarchical Causal Models (ST-HCMs), a novel graphical framework that extends hierarchical causal modeling to the spatio-temporal domain.
- **Methodology:** We validate the effectiveness of our framework on both synthetic and real-world datasets, demonstrating its potential for robust causal inference in complex dynamic systems..
- **Key Finding:** This theoretical result enables a general procedure for causal identification, allowing ST-HCMs to recover causal effects even in the presence of unobserved, time-invariant unit-level confounders, a scenario where standard non-hierarchical models fail.

#### Technical Context
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
* **Limits:** However, inferring causal relationships from such observational data remains challenging, particularly due to unobserved confounders that are specific to units (e.
* **Signal Tags:** #ai #research

---


### Comparative Analysis of Large Language Model Inference Serving Systems: A Performance Study of vLLM and HuggingFace TGI
**Date:** 2025-11-25 | **Arxiv:** [2511.17593](https://arxiv.org/abs/2511.17593)

#### Abstract
The deployment of Large Language Models (LLMs) in production environments requires efficient inference serving systems that balance throughput, latency, and resource utilization. This paper presents a comprehensive empirical evaluation of two prominent open-source LLM serving frameworks: vLLM and HuggingFace Text Generation Inference (TGI). We benchmark these systems across multiple dimensions including throughput performance, end-to-end latency, GPU memory utilization, and scalability characteristics using LLaMA-2 models ranging from 7B to 70B parameters. Our experiments reveal that vLLM achieves up to 24x higher throughput than TGI under high-concurrency workloads through its novel PagedAttention mechanism, while TGI demonstrates lower tail latencies for interactive single-user scenarios. We provide detailed performance profiles for different deployment scenarios and offer practical recommendations for system selection based on workload characteristics. Our findings indicate that the choice between these frameworks should be guided by specific use-case requirements: vLLM excels in high-throughput batch processing scenarios, while TGI is better suited for latency-sensitive interactive applications with moderate concurrency.

#### Research Highlights
- **Core Innovation:** The deployment of Large Language Models (LLMs) in production environments requires efficient inference serving systems that balance throughput, latency, and resource utilization.
- **Methodology:** Our findings indicate that the choice between these frameworks should be guided by specific use-case requirements: vLLM excels in high-throughput batch processing scenarios, while TGI is better suited for latency-sensitive interactive applications with moderate concurrency..
- **Key Finding:** Our experiments reveal that vLLM achieves up to 24x higher throughput than TGI under high-concurrency workloads through its novel PagedAttention mechanism, while TGI demonstrates lower tail latencies for interactive single-user scenarios.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Health system learning achieves generalist neuroimaging models
**Date:** 2025-11-25 | **Arxiv:** [2511.18640](https://arxiv.org/abs/2511.18640)

#### Abstract
Frontier artificial intelligence (AI) models, such as OpenAI's GPT-5 and Meta's DINOv3, have advanced rapidly through training on internet-scale public data, yet such systems lack access to private clinical data. Neuroimaging, in particular, is underrepresented in the public domain due to identifiable facial features within MRI and CT scans, fundamentally restricting model performance in clinical medicine. Here, we show that frontier models underperform on neuroimaging tasks and that learning directly from uncurated data generated during routine clinical care at health systems, a paradigm we call health system learning, yields high-performance, generalist neuroimaging models. We introduce NeuroVFM, a visual foundation model trained on 5.24 million clinical MRI and CT volumes using a scalable volumetric joint-embedding predictive architecture. NeuroVFM learns comprehensive representations of brain anatomy and pathology, achieving state-of-the-art performance across multiple clinical tasks, including radiologic diagnosis and report generation. The model exhibits emergent neuroanatomic understanding and interpretable visual grounding of diagnostic findings. When paired with open-source language models through lightweight visual instruction tuning, NeuroVFM generates radiology reports that surpass frontier models in accuracy, clinical triage, and expert preference. Through clinically grounded visual understanding, NeuroVFM reduces hallucinated findings and critical errors, offering safer clinical decision support. These results establish health system learning as a paradigm for building generalist medical AI and provide a scalable framework for clinical foundation models.

#### Research Highlights
- **Core Innovation:** We introduce NeuroVFM, a visual foundation model trained on 5.24 million clinical MRI and CT volumes using a scalable volumetric joint-embedding predictive architecture.
- **Methodology:** These results establish health system learning as a paradigm for building generalist medical AI and provide a scalable framework for clinical foundation models..
- **Key Finding:** These results establish health system learning as a paradigm for building generalist medical AI and provide a scalable framework for clinical foundation models..

#### Technical Context
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


### Understanding Task Transfer in Vision-Language Models
**Date:** 2025-11-25 | **Arxiv:** [2511.18787](https://arxiv.org/abs/2511.18787)

#### Abstract
Vision-Language Models (VLMs) perform well on multimodal benchmarks but lag behind humans and specialized models on visual perception tasks like depth estimation or object counting. Finetuning on one task can unpredictably affect performance on others, making task-specific finetuning challenging. In this paper, we address this challenge through a systematic study of task transferability. We examine how finetuning a VLM on one perception task affects its zero-shot performance on others. To quantify these effects, we introduce Perfection Gap Factor (PGF), a metric that captures both the breadth and magnitude of transfer. Using three open-weight VLMs evaluated across 13 perception tasks, we construct a task-transfer graph that reveals previously unobserved relationships among perception tasks. Our analysis uncovers patterns of positive and negative transfer, identifies groups of tasks that mutually influence each other, organizes tasks into personas based on their transfer behavior and demonstrates how PGF can guide data selection for more efficient training. These findings highlight both opportunities for positive transfer and risks of negative interference, offering actionable guidance for advancing VLMs.

#### Research Highlights
- **Core Innovation:** To quantify these effects, we introduce Perfection Gap Factor (PGF), a metric that captures both the breadth and magnitude of transfer.
- **Methodology:** Using three open-weight VLMs evaluated across 13 perception tasks, we construct a task-transfer graph that reveals previously unobserved relationships among perception tasks.
- **Key Finding:** Our analysis uncovers patterns of positive and negative transfer, identifies groups of tasks that mutually influence each other, organizes tasks into personas based on their transfer behavior and demonstrates how PGF can guide data selection for more efficient training.

#### Technical Context
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
* **Limits:** challenge through a systematic study of task transferability.
* **Signal Tags:** #ai #research

---


### Learning to Drive Anywhere with Model-Based Reannotation
**Date:** 2025-11-25 | **Arxiv:** [2505.05592](https://arxiv.org/abs/2505.05592)

#### Abstract
Developing broadly generalizable visual navigation policies for robots is a significant challenge, primarily constrained by the availability of large-scale, diverse training data. While curated datasets collected by researchers offer high quality, their limited size restricts policy generalization. To overcome this, we explore leveraging abundant, passively collected data sources, including large volumes of crowd-sourced teleoperation data and unlabeled YouTube videos, despite their potential for lower quality or missing action labels. We propose Model-Based ReAnnotation (MBRA), a framework that utilizes a learned short-horizon, model-based expert model to relabel or generate high-quality actions for these passive datasets. This relabeled data is then distilled into LogoNav, a long-horizon navigation policy conditioned on visual goals or GPS waypoints. We demonstrate that LogoNav, trained using MBRA-processed data, achieves state-of-the-art performance, enabling robust navigation over distances exceeding 300 meters in previously unseen indoor and outdoor environments. Our extensive real-world evaluations, conducted across a fleet of robots (including quadrupeds) in six cities on three continents, validate the policy's ability to generalize and navigate effectively even amidst pedestrians in crowded settings.

#### Research Highlights
- **Core Innovation:** We propose Model-Based ReAnnotation (MBRA), a framework that utilizes a learned short-horizon, model-based expert model to relabel or generate high-quality actions for these passive datasets.
- **Methodology:** We demonstrate that LogoNav, trained using MBRA-processed data, achieves state-of-the-art performance, enabling robust navigation over distances exceeding 300 meters in previously unseen indoor and outdoor environments.
- **Key Finding:** We demonstrate that LogoNav, trained using MBRA-processed data, achieves state-of-the-art performance, enabling robust navigation over distances exceeding 300 meters in previously unseen indoor and outdoor environments.

#### Technical Context
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
* **Limits:** challenge, primarily constrained by the availability of large-scale, diverse training data.
* **Signal Tags:** #ai #research

---


### Statistical physics analysis of graph neural networks: Approaching optimality in the contextual stochastic block model
**Date:** 2025-11-24 | **Arxiv:** [2503.01361](https://arxiv.org/abs/2503.01361)

#### Abstract
Graph neural networks (GNNs) are designed to process data associated with graphs. They are finding an increasing range of applications; however, as with other modern machine learning techniques, their theoretical understanding is limited. GNNs can encounter difficulties in gathering information from nodes that are far apart by iterated aggregation steps. This situation is partly caused by so-called oversmoothing; and overcoming it is one of the practically motivated challenges. We consider the situation where information is aggregated by multiple steps of convolution, leading to graph convolutional networks (GCNs). We analyze the generalization performance of a basic GCN, trained for node classification on data generated by the contextual stochastic block model. We predict its asymptotic performance by deriving the free energy of the problem, using the replica method, in the high-dimensional limit. Calling depth the number of convolutional steps, we show the importance of going to large depth to approach the Bayes-optimality. We detail how the architecture of the GCN has to scale with the depth to avoid oversmoothing. The resulting large depth limit can be close to the Bayes-optimality and leads to a continuous GCN. Technically, we tackle this continuous limit via an approach that resembles dynamical mean-field theory (DMFT) with constraints at the initial and final times. An expansion around large regularization allows us to solve the corresponding equations for the performance of the deep GCN. This promising tool may contribute to the analysis of further deep neural networks.

#### Research Highlights
- **Core Innovation:** Graph neural networks (GNNs) are designed to process data associated with graphs.
- **Methodology:** Technically, we tackle this continuous limit via an approach that resembles dynamical mean-field theory (DMFT) with constraints at the initial and final times.
- **Key Finding:** The resulting large depth limit can be close to the Bayes-optimality and leads to a continuous GCN.

#### Technical Context
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
* **Limits:** however, as with other modern machine learning techniques, their theoretical understanding is limited.
* **Signal Tags:** #ai #research

---
