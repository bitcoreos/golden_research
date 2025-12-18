# Vol 03 General Architecture Research
*Enriched by BITCOREOS | Phase 4 Batch 1*

---

### VisPlay: Self-Evolving Vision-Language Models from Images
**Date:** 2025-11-20 | **Arxiv:** [2511.15661](https://hub.bitwiki.org/t/visplay-self-evolving-vision-language-models-from-images/24861)

#### Abstract
Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at https://bruno686.github.io/VisPlay/

#### Research Highlights
- **Core Innovation:** We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data.
- **Methodology:** We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data.
- **Key Finding:** The project page is available at https://bruno686.github.io/VisPlay/.

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
* **Limits:** However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale.
* **Signal Tags:** #ai #research

---


### AdamHD: Decoupled Huber Decay Regularization for Language Model Pre-Training
**Date:** 2025-11-19 | **Arxiv:** [2511.14721](https://hub.bitwiki.org/t/adamhd-decoupled-huber-decay-regularization-for-language-model-pre-training/24596)

#### Abstract
Adaptive optimizers with decoupled weight decay, such as AdamW, are the de facto standard for pre-training large transformer-based generative models. Yet the quadratic nature of the $\ell_2$ penalty embedded in weight decay drives all parameters toward the origin at the same rate, making the update vulnerable to rare but extreme gradient directions and often over-penalizing well-conditioned coordinates. We propose AdamHuberDecay, a drop-in replacement for AdamW that substitutes the $\ell_2$ penalty with a decoupled smooth Huber regularizer. The resulting update decays parameters quadratically while their magnitude remains below a threshold $δ$, and linearly ($\ell_1$-like) once they exceed $δ$, yielding (i) bounded regularization gradients, (ii) invariance to per-coordinate second-moment rescaling, and (iii) stronger sparsity pressure on overgrown weights.   We derive the closed-form decoupled Huber decay step and show how to integrate it with any Adam-family optimizer at $O(1)$ extra cost. Extensive experiments on GPT-2 and GPT-3 pre-training demonstrate that AdamHuberDecay (a) converges 10-15% faster in wall-clock time, (b) reduces validation perplexity by up to 4 points, (c) delivers performance improvements of 2.5-4.7% across downstream tasks, and (d) yields visibly sparser weight histograms that translate into 20-30% memory savings after magnitude pruning, without tuning the decay coefficient beyond the default grid used for AdamW. Ablations confirm robustness to outlier gradients and large-batch regimes, together with theoretical analyses that bound the expected parameter norm under noisy updates. AdamHuberDecay therefore provides a simple, principled path toward more efficient and resilient training of next-generation foundational generative transformers.

#### Research Highlights
- **Core Innovation:** We propose AdamHuberDecay, a drop-in replacement for AdamW that substitutes the $\ell_2$ penalty with a decoupled smooth Huber regularizer.
- **Methodology:** See abstract.
- **Key Finding:** Extensive experiments on GPT-2 and GPT-3 pre-training demonstrate that AdamHuberDecay (a) converges 10-15% faster in wall-clock time, (b) reduces validation perplexity by up to 4 points, (c) delivers performance improvements of 2.5-4.7% across downstream tasks, and (d) yields visibly sparser weight histograms that translate into 20-30% memory savings after magnitude pruning, without tuning the decay coefficient beyond the default grid used for AdamW.

#### Technical Context
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
* **Limits:** remains below a threshold $δ$, and linearly ($\ell_1$-like) once they exceed $δ$, yielding (i) bounded regularization gradients, (ii) invariance to per-coordinate second-moment rescaling, and (iii) stronger sparsity pressure on overgrown weights.
* **Signal Tags:** #ai #research

---


### LAUD: Integrating Large Language Models with Active Learning for Unlabeled Data
**Date:** 2025-11-19 | **Arxiv:** [2511.14738](https://hub.bitwiki.org/t/laud-integrating-large-language-models-with-active-learning-for-unlabeled-data/24597)

#### Abstract
Large language models (LLMs) have shown a remarkable ability to generalize beyond their pre-training data, and fine-tuning LLMs can elevate performance to human-level and beyond. However, in real-world scenarios, lacking labeled data often prevents practitioners from obtaining well-performing models, thereby forcing practitioners to highly rely on prompt-based approaches that are often tedious, inefficient, and driven by trial and error. To alleviate this issue of lacking labeled data, we present a learning framework integrating LLMs with active learning for unlabeled dataset (LAUD). LAUD mitigates the cold-start problem by constructing an initial label set with zero-shot learning. Experimental results show that LLMs derived from LAUD outperform LLMs with zero-shot or few-shot learning on commodity name classification tasks, demonstrating the effectiveness of LAUD.

#### Research Highlights
- **Core Innovation:** Large language models (LLMs) have shown a remarkable ability to generalize beyond their pre-training data, and fine-tuning LLMs can elevate performance to human-level and beyond.
- **Methodology:** To alleviate this issue of lacking labeled data, we present a learning framework integrating LLMs with active learning for unlabeled dataset (LAUD).
- **Key Finding:** Experimental results show that LLMs derived from LAUD outperform LLMs with zero-shot or few-shot learning on commodity name classification tasks, demonstrating the effectiveness of LAUD..

#### Technical Context
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
* **Limits:** However, in real-world scenarios, lacking labeled data often prevents practitioners from obtaining well-performing models, thereby forcing practitioners to highly rely on prompt-based approaches that are often tedious, inefficient, and driven by trial and error.
* **Signal Tags:** #ai #research

---


### Group-Aware Reinforcement Learning for Output Diversity in Large Language Models
**Date:** 2025-11-18 | **Arxiv:** [2511.12596](https://hub.bitwiki.org/t/group-aware-reinforcement-learning-for-output-diversity-in-large-language-models/24318)

#### Abstract
Large Language Models (LLMs) often suffer from mode collapse, repeatedly generating the same few completions even when many valid answers exist, limiting their diversity across a wide range of tasks. We introduce Group-Aware Policy Optimization (GAPO), a simple extension of the recent and popular Group Relative Policy Optimization (GRPO) that computes rewards over the group as a whole. GAPO enables learning from the group-level properties such as diversity and coverage. We demonstrate GAPO using a frequency-aware reward function that encourages uniform sampling over valid LLM completions, and show that GAPO-trained models produce valid and more diverse model responses. Beyond this setup, GAPO generalizes to open-ended prompts and improves response diversity without compromising accuracy on standard LLM benchmarks (GSM8K, MATH, HumanEval, MMLU-Pro). Our code will be made publicly available.

#### Research Highlights
- **Core Innovation:** We introduce Group-Aware Policy Optimization (GAPO), a simple extension of the recent and popular Group Relative Policy Optimization (GRPO) that computes rewards over the group as a whole.
- **Methodology:** We demonstrate GAPO using a frequency-aware reward function that encourages uniform sampling over valid LLM completions, and show that GAPO-trained models produce valid and more diverse model responses.
- **Key Finding:** We demonstrate GAPO using a frequency-aware reward function that encourages uniform sampling over valid LLM completions, and show that GAPO-trained models produce valid and more diverse model responses.

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


### Neuro-Logic Lifelong Learning
**Date:** 2025-11-18 | **Arxiv:** [2511.12793](https://hub.bitwiki.org/t/neuro-logic-lifelong-learning/24333)

#### Abstract
Solving Inductive Logic Programming (ILP) problems with neural networks is a key challenge in Neural-Symbolic Ar- tificial Intelligence (AI). While most research has focused on designing novel network architectures for individual prob- lems, less effort has been devoted to exploring new learning paradigms involving a sequence of problems. In this work, we investigate lifelong learning ILP, which leverages the com- positional and transferable nature of logic rules for efficient learning of new problems. We introduce a compositional framework, demonstrating how logic rules acquired from ear- lier tasks can be efficiently reused in subsequent ones, leading to improved scalability and performance. We formalize our approach and empirically evaluate it on sequences of tasks. Experimental results validate the feasibility and advantages of this paradigm, opening new directions for continual learn- ing in Neural-Symbolic AI.

#### Research Highlights
- **Core Innovation:** We introduce a compositional framework, demonstrating how logic rules acquired from ear- lier tasks can be efficiently reused in subsequent ones, leading to improved scalability and performance.
- **Methodology:** We introduce a compositional framework, demonstrating how logic rules acquired from ear- lier tasks can be efficiently reused in subsequent ones, leading to improved scalability and performance.
- **Key Finding:** Experimental results validate the feasibility and advantages of this paradigm, opening new directions for continual learn- ing in Neural-Symbolic AI..

#### Technical Context
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
* **Limits:** challenge in Neural-Symbolic Ar- tificial Intelligence (AI).
* **Signal Tags:** #ai #research

---


### NeuralOM: Neural Ocean Model for Subseasonal-to-Seasonal Simulation
**Date:** 2025-11-18 | **Arxiv:** [2505.21020](https://hub.bitwiki.org/t/neuralom-neural-ocean-model-for-subseasonal-to-seasonal-simulation/24419)

#### Abstract
Long-term, high-fidelity simulation of slow-changing physical systems, such as the ocean and climate, presents a fundamental challenge in scientific computing. Traditional autoregressive machine learning models often fail in these tasks as minor errors accumulate and lead to rapid forecast degradation. To address this problem, we propose NeuralOM, a general neural operator framework designed for simulating complex, slow-changing dynamics. NeuralOM's core consists of two key innovations: (1) a Progressive Residual Correction Framework that decomposes the forecasting task into a series of fine-grained refinement steps, effectively suppressing long-term error accumulation; and (2) a Physics-Guided Graph Network whose built-in adaptive messaging mechanism explicitly models multi-scale physical interactions, such as gradient-driven flows and multiplicative couplings, thereby enhancing physical consistency while maintaining computational efficiency. We validate NeuralOM on the challenging task of global Subseasonal-to-Seasonal (S2S) ocean simulation. Extensive experiments demonstrate that NeuralOM not only surpasses state-of-the-art models in forecast accuracy and long-term stability, but also excels in simulating extreme events. For instance, at a 60-day lead time, NeuralOM achieves a 13.3% lower RMSE compared to the best-performing baseline, offering a stable, efficient, and physically-aware paradigm for data-driven scientific computing. Code link: https://github.com/YuanGao-YG/NeuralOM.

#### Research Highlights
- **Core Innovation:** To address this problem, we propose NeuralOM, a general neural operator framework designed for simulating complex, slow-changing dynamics.
- **Methodology:** NeuralOM's core consists of two key innovations: (1) a Progressive Residual Correction Framework that decomposes the forecasting task into a series of fine-grained refinement steps, effectively suppressing long-term error accumulation; and (2) a Physics-Guided Graph Network whose built-in adaptive messaging mechanism explicitly models multi-scale physical interactions, such as gradient-driven flows and multiplicative couplings, thereby enhancing physical consistency while maintaining computational efficiency.
- **Key Finding:** Extensive experiments demonstrate that NeuralOM not only surpasses state-of-the-art models in forecast accuracy and long-term stability, but also excels in simulating extreme events.

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
* **Limits:** challenge in scientific computing.
* **Signal Tags:** #ai #research

---


### Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting
**Date:** 2025-11-18 | **Arxiv:** [2511.13052](https://hub.bitwiki.org/t/learning-from-the-undesirable-robust-adaptation-of-language-models-without-forgetting/24183)

#### Abstract
Language models (LMs) are often adapted through supervised fine-tuning (SFT) to specialize their capabilities for downstream tasks. However, in typical scenarios where the fine-tuning data is limited, e.g., compared to pre-training, SFT can lead LMs to overfit, causing them to rely on spurious patterns within the target task or to compromise other broadly useful capabilities as a side effect of narrow specialization. In this paper, we propose Learning-from-the-Undesirable (LfU), a simple yet effective regularization scheme for SFT to mitigate overfitting issues when fine-tuning LMs with limited data. Specifically, we aim to regularize the fine-tuning process to favor solutions that are resilient to "undesirable" model updates, e.g., gradient ascent steps that steer the model toward undesirable behaviors. To this end, we propose a novel form of consistency regularization that directly aligns internal representations of the model with those after an undesirable update. By leveraging representation-level data augmentation through undesirable updates, LfU effectively promotes generalization under limited data. Our experiments on diverse LM downstream tasks show that LfU serves as an effective prior that enhances adaptability while preserving pretrained knowledge. For example, our LM from LfU achieves a 16.8% average improvement on math tasks compared to vanilla SFT on the same dataset, where the latter even leads to degraded performance on those tasks. Furthermore, LfU exhibits improved robustness to prompt variations, e.g., yielding a 92.1% lower standard deviation in output performances compared to SFT, highlighting its versatile effects.

#### Research Highlights
- **Core Innovation:** To this end, we propose a novel form of consistency regularization that directly aligns internal representations of the model with those after an undesirable update.
- **Methodology:** Furthermore, LfU exhibits improved robustness to prompt variations, e.g., yielding a 92.1% lower standard deviation in output performances compared to SFT, highlighting its versatile effects..
- **Key Finding:** Our experiments on diverse LM downstream tasks show that LfU serves as an effective prior that enhances adaptability while preserving pretrained knowledge.

#### Technical Context
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
* **Limits:** However, in typical scenarios where the fine-tuning data is limited, e.
* **Signal Tags:** #ai #research

---


### The Good, The Bad, and The Hybrid: A Reward Structure Showdown in Reasoning Models Training
**Date:** 2025-11-18 | **Arxiv:** [2511.13016](https://hub.bitwiki.org/t/the-good-the-bad-and-the-hybrid-a-reward-structure-showdown-in-reasoning-models-training/24177)

#### Abstract
Reward design is central to reinforcement learning from human feedback (RLHF) and alignment research. In this work, we propose a unified framework to study hard, continuous, and hybrid reward structures for fine-tuning large language models (LLMs) on mathematical reasoning tasks. Using Qwen3-4B with LoRA fine-tuning on the GSM8K dataset, we formalize and empirically evaluate reward formulations that incorporate correctness, perplexity, reasoning quality, and consistency. We introduce an adaptive hybrid reward scheduler that transitions between discrete and continuous signals, balancing exploration and stability. Our results show that hybrid reward structures improve convergence speed and training stability over purely hard or continuous approaches, offering insights for alignment via adaptive reward modeling.

#### Research Highlights
- **Core Innovation:** We introduce an adaptive hybrid reward scheduler that transitions between discrete and continuous signals, balancing exploration and stability.
- **Methodology:** Our results show that hybrid reward structures improve convergence speed and training stability over purely hard or continuous approaches, offering insights for alignment via adaptive reward modeling..
- **Key Finding:** Our results show that hybrid reward structures improve convergence speed and training stability over purely hard or continuous approaches, offering insights for alignment via adaptive reward modeling..

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Scaling Law Analysis in Federated Learning: How to Select the Optimal Model Size?
**Date:** 2025-11-18 | **Arxiv:** [2511.12188](https://hub.bitwiki.org/t/scaling-law-analysis-in-federated-learning-how-to-select-the-optimal-model-size/24096)

#### Abstract
The recent success of large language models (LLMs) has sparked a growing interest in training large-scale models. As the model size continues to scale, concerns are growing about the depletion of high-quality, well-curated training data. This has led practitioners to explore training approaches like Federated Learning (FL), which can leverage the abundant data on edge devices while maintaining privacy. However, the decentralization of training datasets in FL introduces challenges to scaling large models, a topic that remains under-explored. This paper fills this gap and provides qualitative insights on generalizing the previous model scaling experience to federated learning scenarios. Specifically, we derive a PAC-Bayes (Probably Approximately Correct Bayesian) upper bound for the generalization error of models trained with stochastic algorithms in federated settings and quantify the impact of distributed training data on the optimal model size by finding the analytic solution of model size that minimizes this bound. Our theoretical results demonstrate that the optimal model size has a negative power law relationship with the number of clients if the total training compute is unchanged. Besides, we also find that switching to FL with the same training compute will inevitably reduce the upper bound of generalization performance that the model can achieve through training, and that estimating the optimal model size in federated scenarios should depend on the average training compute across clients. Furthermore, we also empirically validate the correctness of our results with extensive training runs on different models, network settings, and datasets.

#### Research Highlights
- **Core Innovation:** However, the decentralization of training datasets in FL introduces challenges to scaling large models, a topic that remains under-explored.
- **Methodology:** See abstract.
- **Key Finding:** Furthermore, we also empirically validate the correctness of our results with extensive training runs on different models, network settings, and datasets..

#### Technical Context
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
* **Limits:** However, the decentralization of training datasets in FL introduces challenges to scaling large models, a topic that remains under-explored.
* **Signal Tags:** #ai #research

---


### Practical Causal Evaluation Metrics for Biological Networks
**Date:** 2025-11-18 | **Arxiv:** [2511.12805](https://hub.bitwiki.org/t/practical-causal-evaluation-metrics-for-biological-networks/24334)

#### Abstract
Estimating causal networks from biological data is a critical step in systems biology. When evaluating the inferred network, assessing the networks based on their intervention effects is particularly important for downstream probabilistic reasoning and the identification of potential drug targets. In the context of gene regulatory network inference, biological databases are often used as reference sources. These databases typically describe relationships in a qualitative rather than quantitative manner. However, few evaluation metrics have been developed that take this qualitative nature into account. To address this, we developed a metric, the sign-augmented Structural Intervention Distance (sSID), and a weighted sSID that incorporates the net effects of the intervention. Through simulations and analyses of real transcriptomic datasets, we found that our proposed metrics could identify a different algorithm as optimal compared to conventional metrics, and the network selected by sSID had a superior performance in the classification task of clinical covariates using transcriptomic data. This suggests that sSID can distinguish networks that are structurally correct but functionally incorrect, highlighting its potential as a more biologically meaningful and practical evaluation metric.

#### Research Highlights
- **Core Innovation:** Through simulations and analyses of real transcriptomic datasets, we found that our proposed metrics could identify a different algorithm as optimal compared to conventional metrics, and the network selected by sSID had a superior performance in the classification task of clinical covariates using transcriptomic data.
- **Methodology:** Through simulations and analyses of real transcriptomic datasets, we found that our proposed metrics could identify a different algorithm as optimal compared to conventional metrics, and the network selected by sSID had a superior performance in the classification task of clinical covariates using transcriptomic data.
- **Key Finding:** This suggests that sSID can distinguish networks that are structurally correct but functionally incorrect, highlighting its potential as a more biologically meaningful and practical evaluation metric..

#### Technical Context
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
* **Limits:** However, few evaluation metrics have been developed that take this qualitative nature into account.
* **Signal Tags:** #ai #research

---


### Doubly Debiased Test-Time Prompt Tuning for Vision-Language Models
**Date:** 2025-11-18 | **Arxiv:** [2511.11690](https://hub.bitwiki.org/t/doubly-debiased-test-time-prompt-tuning-for-vision-language-models/24004)

#### Abstract
Test-time prompt tuning for vision-language models has demonstrated impressive generalization capabilities under zero-shot settings. However, tuning the learnable prompts solely based on unlabeled test data may induce prompt optimization bias, ultimately leading to suboptimal performance on downstream tasks. In this work, we analyze the underlying causes of prompt optimization bias from both the model and data perspectives. In terms of the model, the entropy minimization objective typically focuses on reducing the entropy of model predictions while overlooking their correctness. This can result in overconfident yet incorrect outputs, thereby compromising the quality of prompt optimization. On the data side, prompts affected by optimization bias can introduce misalignment between visual and textual modalities, which further aggravates the prompt optimization bias. To this end, we propose a Doubly Debiased Test-Time Prompt Tuning method. Specifically, we first introduce a dynamic retrieval-augmented modulation module that retrieves high-confidence knowledge from a dynamic knowledge base using the test image feature as a query, and uses the retrieved knowledge to modulate the predictions. Guided by the refined predictions, we further develop a reliability-aware prompt optimization module that incorporates a confidence-based weighted ensemble and cross-modal consistency distillation to impose regularization constraints during prompt tuning. Extensive experiments across 15 benchmark datasets involving both natural distribution shifts and cross-datasets generalization demonstrate that our method outperforms baselines, validating its effectiveness in mitigating prompt optimization bias.

#### Research Highlights
- **Core Innovation:** Specifically, we first introduce a dynamic retrieval-augmented modulation module that retrieves high-confidence knowledge from a dynamic knowledge base using the test image feature as a query, and uses the retrieved knowledge to modulate the predictions.
- **Methodology:** Specifically, we first introduce a dynamic retrieval-augmented modulation module that retrieves high-confidence knowledge from a dynamic knowledge base using the test image feature as a query, and uses the retrieved knowledge to modulate the predictions.
- **Key Finding:** Extensive experiments across 15 benchmark datasets involving both natural distribution shifts and cross-datasets generalization demonstrate that our method outperforms baselines, validating its effectiveness in mitigating prompt optimization bias..

#### Technical Context
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
* **Limits:** However, tuning the learnable prompts solely based on unlabeled test data may induce prompt optimization bias, ultimately leading to suboptimal performance on downstream tasks.
* **Signal Tags:** #ai #research

---


### A Deep Learning Model to Predicting Changes in Consumer Attributes for New Line-extended Products
**Date:** 2025-11-18 | **Arxiv:** [2511.11646](https://hub.bitwiki.org/t/a-deep-learning-model-to-predicting-changes-in-consumer-attributes-for-new-line-extended-products/23952)

#### Abstract
Product line extension is a marketing strategy that enhances a company's sphere of influence. Because excessive line extensions disrupt brand image, only appropriate line extensions based on consumer needs are desirable. Marketers should know the key consumer attributes of the primary customers for new line-extended products before companies enter the market. This paper describes a method for predicting changes in consumer attributes for new line-extended products using a novel deep learning model. The proposed model, Conditional Tabular Variational Auto-Encoder (CTVAE), generates synthetic data from large-scale tabular data of consumers and products. It can provide various implications about effective product line marketing for marketers. The experimental results demonstrate that the CTVAE offers superior prediction performance than existing models. We indicate implications for new products that change containers or flavors for effective product line marketing. The proposed approach has the potential to contribute to avoiding cannibalization and to designing product images and marketing strategies.

#### Research Highlights
- **Core Innovation:** The proposed approach has the potential to contribute to avoiding cannibalization and to designing product images and marketing strategies..
- **Methodology:** This paper describes a method for predicting changes in consumer attributes for new line-extended products using a novel deep learning model.
- **Key Finding:** The experimental results demonstrate that the CTVAE offers superior prediction performance than existing models.

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


### Foundations of Structural Causal Models with Latent Selection
**Date:** 2025-11-18 | **Arxiv:** [2401.06925](https://hub.bitwiki.org/t/foundations-of-structural-causal-models-with-latent-selection/24025)

#### Abstract
Three distinct phenomena complicate statistical causal analysis: latent common causes, causal cycles, and latent selection. Foundational works on Structural Causal Models (SCMs), e.g., Bongers et al. (2021, Ann. Stat., 49(5): 2885-2915), treat cycles and latent variables, while an analogous account of latent selection is missing. The goal of this article is to develop a theoretical foundation for modeling latent selection with SCMs. To achieve that, we introduce a conditioning operation for SCMs: it maps an SCM with explicit selection mechanisms to one without them while preserving the causal semantics of the selected subpopulation. Graphically, in Directed Mixed Graphs we extend bidirected edge--beyond latent common cause--to also encode latent selection. We prove that the conditioning operation preserves simplicity, acyclicity, and linearity of SCMs, and interacts well with marginalization, conditioning, and interventions. These properties make those three operations valuable tools for causal modeling, reasoning, and learning after abstracting away latent details (latent common causes and selection). Examples show how this abstraction streamlines analysis and clarifies when standard tools (e.g., adjustment, causal calculus, instrumental variables) remain valid under selection bias. We hope that these results deepen the SCM-based understanding of selection bias and become part of the standard causal modeling toolbox to build more reliable causal analysis.

#### Research Highlights
- **Core Innovation:** To achieve that, we introduce a conditioning operation for SCMs: it maps an SCM with explicit selection mechanisms to one without them while preserving the causal semantics of the selected subpopulation.
- **Methodology:** See abstract.
- **Key Finding:** We hope that these results deepen the SCM-based understanding of selection bias and become part of the standard causal modeling toolbox to build more reliable causal analysis..

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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments
**Date:** 2025-11-18 | **Arxiv:** [2505.19361](https://hub.bitwiki.org/t/consistency-based-abductive-reasoning-over-perceptual-errors-of-multiple-pre-trained-models-in-novel-environments/24460)

#### Abstract
The deployment of pre-trained perception models in novel environments often leads to performance degradation due to distributional shifts. Although recent artificial intelligence approaches for metacognition use logical rules to characterize and filter model errors, improving precision often comes at the cost of reduced recall. This paper addresses the hypothesis that leveraging multiple pre-trained models can mitigate this recall reduction. We formulate the challenge of identifying and managing conflicting predictions from various models as a consistency-based abduction problem, building on the idea of abductive learning (ABL) but applying it to test-time instead of training. The input predictions and the learned error detection rules derived from each model are encoded in a logic program. We then seek an abductive explanation--a subset of model predictions--that maximizes prediction coverage while ensuring the rate of logical inconsistencies (derived from domain constraints) remains below a specified threshold. We propose two algorithms for this knowledge representation task: an exact method based on Integer Programming (IP) and an efficient Heuristic Search (HS). Through extensive experiments on a simulated aerial imagery dataset featuring controlled, complex distributional shifts, we demonstrate that our abduction-based framework outperforms individual models and standard ensemble baselines, achieving, for instance, average relative improvements of approximately 13.6\% in F1-score and 16.6\% in accuracy across 15 diverse test datasets when compared to the best individual model. Our results validate the use of consistency-based abduction as an effective mechanism to robustly integrate knowledge from multiple imperfect models in challenging, novel scenarios.

#### Research Highlights
- **Core Innovation:** We propose two algorithms for this knowledge representation task: an exact method based on Integer Programming (IP) and an efficient Heuristic Search (HS).
- **Methodology:** Through extensive experiments on a simulated aerial imagery dataset featuring controlled, complex distributional shifts, we demonstrate that our abduction-based framework outperforms individual models and standard ensemble baselines, achieving, for instance, average relative improvements of approximately 13.6\% in F1-score and 16.6\% in accuracy across 15 diverse test datasets when compared to the best individual model.
- **Key Finding:** Our results validate the use of consistency-based abduction as an effective mechanism to robustly integrate knowledge from multiple imperfect models in challenging, novel scenarios..

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
* **Limits:** Although recent artificial intelligence approaches for metacognition use logical rules to characterize and filter model errors, improving precision often comes at the cost of reduced recall.
* **Signal Tags:** #ai #research

---


### Towards Uncertainty Quantification in Generative Model Learning
**Date:** 2025-11-17 | **Arxiv:** [2511.10710](https://hub.bitwiki.org/t/towards-uncertainty-quantification-in-generative-model-learning/23689)

#### Abstract
While generative models have become increasingly prevalent across various domains, fundamental concerns regarding their reliability persist. A crucial yet understudied aspect of these models is the uncertainty quantification surrounding their distribution approximation capabilities. Current evaluation methodologies focus predominantly on measuring the closeness between the learned and the target distributions, neglecting the inherent uncertainty in these measurements. In this position paper, we formalize the problem of uncertainty quantification in generative model learning. We discuss potential research directions, including the use of ensemble-based precision-recall curves. Our preliminary experiments on synthetic datasets demonstrate the effectiveness of aggregated precision-recall curves in capturing model approximation uncertainty, enabling systematic comparison among different model architectures based on their uncertainty characteristics.

#### Research Highlights
- **Core Innovation:** While generative models have become increasingly prevalent across various domains, fundamental concerns regarding their reliability persist.
- **Methodology:** See abstract.
- **Key Finding:** Our preliminary experiments on synthetic datasets demonstrate the effectiveness of aggregated precision-recall curves in capturing model approximation uncertainty, enabling systematic comparison among different model architectures based on their uncertainty characteristics..

#### Technical Context
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


### HierRouter: Coordinated Routing of Specialized Large Language Models via Reinforcement Learning
**Date:** 2025-11-14 | **Arxiv:** [2511.09873](https://hub.bitwiki.org/t/hierrouter-coordinated-routing-of-specialized-large-language-models-via-reinforcement-learning/23561)

#### Abstract
Large Language Models (LLMs) deliver state-of-the-art performance across many tasks but impose high computational and memory costs, limiting their deployment in resource-constrained or real-time settings. To address this, we propose HierRouter, a hierarchical routing approach that dynamically assembles inference pipelines from a pool of specialized, lightweight language models. Formulated as a finite-horizon Markov Decision Process (MDP), our approach trains a Proximal Policy Optimization (PPO)-based reinforcement learning agent to iteratively select which models to invoke at each stage of multi-hop inference. The agent conditions on the evolving context and accumulated cost to make context-aware routing decisions. Experiments with three open-source candidate LLMs across six benchmarks, including QA, code generation, and mathematical reasoning, show that HierRouter improves response quality by up to 2.4x compared to using individual models independently, while incurring only a minimal additional inference cost on average. These results highlight the promise of hierarchical routing for cost-efficient, high-performance LLM inference. All codes can be found here https://github.com/ Nikunj-Gupta/hierouter.

#### Research Highlights
- **Core Innovation:** To address this, we propose HierRouter, a hierarchical routing approach that dynamically assembles inference pipelines from a pool of specialized, lightweight language models.
- **Methodology:** Experiments with three open-source candidate LLMs across six benchmarks, including QA, code generation, and mathematical reasoning, show that HierRouter improves response quality by up to 2.4x compared to using individual models independently, while incurring only a minimal additional inference cost on average.
- **Key Finding:** These results highlight the promise of hierarchical routing for cost-efficient, high-performance LLM inference.

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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Completion of partial structures using Patterson maps with the CrysFormer machine learning model
**Date:** 2025-11-14 | **Arxiv:** [2511.10440](https://hub.bitwiki.org/t/completion-of-partial-structures-using-patterson-maps-with-the-crysformer-machine-learning-model/23578)

#### Abstract
Protein structure determination has long been one of the primary challenges of structural biology, to which deep machine learning (ML)-based approaches have increasingly been applied. However, these ML models generally do not incorporate the experimental measurements directly, such as X-ray crystallographic diffraction data. To this end, we explore an approach that more tightly couples these traditional crystallographic and recent ML-based methods, by training a hybrid 3-d vision transformer and convolutional network on inputs from both domains. We make use of two distinct input constructs / Patterson maps, which are directly obtainable from crystallographic data, and ``partial structure'' template maps derived from predicted structures deposited in the AlphaFold Protein Structure Database with subsequently omitted residues. With these, we predict electron density maps that are then post-processed into atomic models through standard crystallographic refinement processes. Introducing an initial dataset of small protein fragments taken from Protein Data Bank entries and placing them in hypothetical crystal settings, we demonstrate that our method is effective at both improving the phases of the crystallographic structure factors and completing the regions missing from partial structure templates, as well as improving the agreement of the electron density maps with the ground truth atomic structures.

#### Research Highlights
- **Core Innovation:** Protein structure determination has long been one of the primary challenges of structural biology, to which deep machine learning (ML)-based approaches have increasingly been applied.
- **Methodology:** See abstract.
- **Key Finding:** Introducing an initial dataset of small protein fragments taken from Protein Data Bank entries and placing them in hypothetical crystal settings, we demonstrate that our method is effective at both improving the phases of the crystallographic structure factors and completing the regions missing from partial structure templates, as well as improving the agreement of the electron density maps with the ground truth atomic structures..

#### Technical Context
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
* **Limits:** However, these ML models generally do not incorporate the experimental measurements directly, such as X-ray crystallographic diffraction data.
* **Signal Tags:** #ai #research

---


### Filtering Jump Markov Systems with Partially Known Dynamics: A Model-Based Deep Learning Approach
**Date:** 2025-11-14 | **Arxiv:** [2511.09569](https://hub.bitwiki.org/t/filtering-jump-markov-systems-with-partially-known-dynamics-a-model-based-deep-learning-approach/23418)

#### Abstract
This paper presents the Jump Markov Filtering Network (JMFNet), a novel model-based deep learning framework for real-time state-state estimation in jump Markov systems with unknown noise statistics and mode transition dynamics. A hybrid architecture comprising two Recurrent Neural Networks (RNNs) is proposed: one for mode prediction and another for filtering that is based on a mode-augmented version of the recently presented KalmanNet architecture. The proposed RNNs are trained jointly using an alternating least squares strategy that enables mutual adaptation without supervision of the latent modes. Extensive numerical experiments on linear and nonlinear systems, including target tracking, pendulum angle tracking, Lorenz attractor dynamics, and a real-life dataset demonstrate that the proposed JMFNet framework outperforms classical model-based filters (e.g., interacting multiple models and particle filters) as well as model-free deep learning baselines, particularly in non-stationary and high-noise regimes. It is also showcased that JMFNet achieves a small yet meaningful improvement over the KalmanNet framework, which becomes much more pronounced in complicated systems or long trajectories. Finally, the method's performance is empirically validated to be consistent and reliable, exhibiting low sensitivity to initial conditions, hyperparameter selection, as well as to incorrect model knowledge

#### Research Highlights
- **Core Innovation:** Extensive numerical experiments on linear and nonlinear systems, including target tracking, pendulum angle tracking, Lorenz attractor dynamics, and a real-life dataset demonstrate that the proposed JMFNet framework outperforms classical model-based filters (e.g., interacting multiple models and particle filters) as well as model-free deep learning baselines, particularly in non-stationary and high-noise regimes.
- **Methodology:** It is also showcased that JMFNet achieves a small yet meaningful improvement over the KalmanNet framework, which becomes much more pronounced in complicated systems or long trajectories.
- **Key Finding:** It is also showcased that JMFNet achieves a small yet meaningful improvement over the KalmanNet framework, which becomes much more pronounced in complicated systems or long trajectories.

#### Technical Context
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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models
**Date:** 2025-11-14 | **Arxiv:** [2511.09809](https://hub.bitwiki.org/t/test-time-spectrum-aware-latent-steering-for-zero-shot-generalization-in-vision-language-models/23560)

#### Abstract
Vision-Language Models (VLMs) excel at zero-shot inference but often degrade under test-time domain shifts. For this reason, episodic test-time adaptation strategies have recently emerged as powerful techniques for adapting VLMs to a single unlabeled image. However, existing adaptation strategies, such as test-time prompt tuning, typically require backpropagating through large encoder weights or altering core model components. In this work, we introduce Spectrum-Aware Test-Time Steering (STS), a lightweight adaptation framework that extracts a spectral subspace from the textual embeddings to define principal semantic directions and learns to steer latent representations in a spectrum-aware manner by adapting a small number of per-sample shift parameters to minimize entropy across augmented views. STS operates entirely at inference in the latent space, without backpropagation through or modification of the frozen encoders. Building on standard evaluation protocols, our comprehensive experiments demonstrate that STS largely surpasses or compares favorably against state-of-the-art test-time adaptation methods, while introducing only a handful of additional parameters and achieving inference speeds up to 8x faster with a 12x smaller memory footprint than conventional test-time prompt tuning. The code is available at https://github.com/kdafnis/STS.

#### Research Highlights
- **Core Innovation:** In this work, we introduce Spectrum-Aware Test-Time Steering (STS), a lightweight adaptation framework that extracts a spectral subspace from the textual embeddings to define principal semantic directions and learns to steer latent representations in a spectrum-aware manner by adapting a small number of per-sample shift parameters to minimize entropy across augmented views.
- **Methodology:** In this work, we introduce Spectrum-Aware Test-Time Steering (STS), a lightweight adaptation framework that extracts a spectral subspace from the textual embeddings to define principal semantic directions and learns to steer latent representations in a spectrum-aware manner by adapting a small number of per-sample shift parameters to minimize entropy across augmented views.
- **Key Finding:** Building on standard evaluation protocols, our comprehensive experiments demonstrate that STS largely surpasses or compares favorably against state-of-the-art test-time adaptation methods, while introducing only a handful of additional parameters and achieving inference speeds up to 8x faster with a 12x smaller memory footprint than conventional test-time prompt tuning.

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
* **Limits:** However, existing adaptation strategies, such as test-time prompt tuning, typically require backpropagating through large encoder weights or altering core model components.
* **Signal Tags:** #ai #research

---


### From Model Training to Model Raising
**Date:** 2025-11-13 | **Arxiv:** [2511.09287](https://hub.bitwiki.org/t/from-model-training-to-model-raising/23336)

#### Abstract
Current AI training methods align models with human values only after their core capabilities have been established, resulting in models that are easily misaligned and lack deep-rooted value systems. We propose a paradigm shift from "model training" to "model raising", in which alignment is woven into a model's development from the start. We identify several key components for this paradigm, all centered around redesigning the training corpus: reframing training data from a first-person perspective, recontextualizing information as lived experience, simulating social interactions, and scaffolding the ordering of training data. We expect that this redesign of the training corpus will lead to an early commitment to values from the first training token onward, such that knowledge, skills, and values are intrinsically much harder to separate. In an ecosystem in which large language model capabilities start overtaking human capabilities in many tasks, this seems to us like a critical need.

#### Research Highlights
- **Core Innovation:** We propose a paradigm shift from "model training" to "model raising", in which alignment is woven into a model's development from the start.
- **Methodology:** See abstract.
- **Key Finding:** Current AI training methods align models with human values only after their core capabilities have been established, resulting in models that are easily misaligned and lack deep-rooted value systems.

#### Technical Context
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


### Kodezi Chronos: A Debugging-First Language Model for Repository-Scale Code Understanding
**Date:** 2025-11-13 | **Arxiv:** [2507.12482](https://hub.bitwiki.org/t/kodezi-chronos-a-debugging-first-language-model-for-repository-scale-code-understanding/23391)

#### Abstract
Large Language Models (LLMs) have advanced code generation and software automation but remain constrained by inference-time context and lack structured reasoning over code, leaving debugging largely unsolved. While Claude 4.5 Opus achieves 74.40% on SWE-bench Verified and Gemini 3 Pro reaches 76.2%, both models remain below 20% on real multi-file debugging tasks. We introduce Kodezi Chronos-1, a language model purpose-built for debugging that integrates Adaptive Graph-Guided Retrieval to navigate codebases up to 10 million lines (92% precision, 85% recall), Persistent Debug Memory trained on over 15 million sessions, and a seven-layer fix-test-refine architecture. On 5,000 real-world scenarios, Chronos-1 achieves 67.3% +/- 2.1% fix accuracy compared to 14.2% +/- 1.3% for Claude 4.1 Opus and 13.8% +/- 1.2% for GPT-4.1 (Cohen's d = 3.87). On SWE-bench Lite, Chronos-1 reaches a state-of-the-art 80.33% resolution rate (241 of 300), outperforming the next best system by 20 points and achieving repository-specific highs of 96.1% on Sympy and 90.4% on Django. Chronos-1 reduces debugging time by 40% and iterations by 65%, resolving complex multi-file and cross-repository bugs that require temporal analysis. Limitations remain for hardware-dependent and dynamic language errors, and Chronos-1 will be available in Kodezi OS in Q4 2025 and via API in Q1 2026.

#### Research Highlights
- **Core Innovation:** We introduce Kodezi Chronos-1, a language model purpose-built for debugging that integrates Adaptive Graph-Guided Retrieval to navigate codebases up to 10 million lines (92% precision, 85% recall), Persistent Debug Memory trained on over 15 million sessions, and a seven-layer fix-test-refine architecture.
- **Methodology:** Limitations remain for hardware-dependent and dynamic language errors, and Chronos-1 will be available in Kodezi OS in Q4 2025 and via API in Q1 2026..
- **Key Finding:** Limitations remain for hardware-dependent and dynamic language errors, and Chronos-1 will be available in Kodezi OS in Q4 2025 and via API in Q1 2026..

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
* **Limits:** Limitations remain for hardware-dependent and dynamic language errors, and Chronos-1 will be available in Kodezi OS in Q4 2025 and via API in Q1 2026.
* **Signal Tags:** #ai #research

---


### How does the Performance of the Data-driven Traffic Flow Forecasting Models deteriorate with Increasing Forecasting Horizon? An Extensive Approach Considering Statistical, Machine Learning and Deep Learning Models
**Date:** 2025-11-13 | **Arxiv:** [2511.09450](https://hub.bitwiki.org/t/how-does-the-performance-of-the-data-driven-traffic-flow-forecasting-models-deteriorate-with-increasing-forecasting-horizon-an-extensive-approach-considering-statistical-machine-learning-and-deep-learning-models/23282)

#### Abstract
With rapid urbanization in recent decades, traffic congestion has intensified due to increased movement of people and goods. As planning shifts from demand-based to supply-oriented strategies, Intelligent Transportation Systems (ITS) have become essential for managing traffic within existing infrastructure. A core ITS function is traffic forecasting, enabling proactive measures like ramp metering, signal control, and dynamic routing through platforms such as Google Maps. This study assesses the performance of statistical, machine learning (ML), and deep learning (DL) models in forecasting traffic speed and flow using real-world data from California's Harbor Freeway, sourced from the Caltrans Performance Measurement System (PeMS). Each model was evaluated over 20 forecasting windows (up to 1 hour 40 minutes) using RMSE, MAE, and R-Square metrics. Results show ANFIS-GP performs best at early windows with RMSE of 0.038, MAE of 0.0276, and R-Square of 0.9983, while Bi-LSTM is more robust for medium-term prediction due to its capacity to model long-range temporal dependencies, achieving RMSE of 0.1863, MAE of 0.0833, and R-Square of 0.987 at a forecasting of 20. The degradation in model performance was quantified using logarithmic transformation, with slope values used to measure robustness. Among DL models, Bi-LSTM had the flattest slope (0.0454 RMSE, 0.0545 MAE for flow), whereas ANFIS-GP had 0.1058 for RMSE and 0.1037 for flow MAE. The study concludes by identifying hybrid models as a promising future direction.

#### Research Highlights
- **Core Innovation:** With rapid urbanization in recent decades, traffic congestion has intensified due to increased movement of people and goods.
- **Methodology:** The degradation in model performance was quantified using logarithmic transformation, with slope values used to measure robustness.
- **Key Finding:** Results show ANFIS-GP performs best at early windows with RMSE of 0.038, MAE of 0.0276, and R-Square of 0.9983, while Bi-LSTM is more robust for medium-term prediction due to its capacity to model long-range temporal dependencies, achieving RMSE of 0.1863, MAE of 0.0833, and R-Square of 0.987 at a forecasting of 20.

#### Technical Context
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


### Foam Segmentation in Wastewater Treatment Plants: A Federated Learning Approach with Segment Anything Model 2
**Date:** 2025-11-12 | **Arxiv:** [2511.08130](https://hub.bitwiki.org/t/foam-segmentation-in-wastewater-treatment-plants-a-federated-learning-approach-with-segment-anything-model-2/23070)

#### Abstract
Foam formation in Wastewater Treatment Plants (WTPs) is a major challenge that can reduce treatment efficiency and increase costs. The ability to automatically examine changes in real-time with respect to the percentage of foam can be of great benefit to the plant. However, large amounts of labeled data are required to train standard Machine Learning (ML) models. The development of these systems is slow due to the scarcity and heterogeneity of labeled data. Additionally, the development is often hindered by the fact that different WTPs do not share their data due to privacy concerns. This paper proposes a new framework to address these challenges by combining Federated Learning (FL) with the state-of-the-art base model for image segmentation, Segment Anything Model 2 (SAM2). The FL paradigm enables collaborative model training across multiple WTPs without centralizing sensitive operational data, thereby ensuring privacy. The framework accelerates training convergence and improves segmentation performance even with limited local datasets by leveraging SAM2's strong pre-trained weights for initialization. The methodology involves fine-tuning SAM2 on distributed clients (edge nodes) using the Flower framework, where a central Fog server orchestrates the process by aggregating model weights without accessing private data. The model was trained and validated using various data collections, including real-world images captured at a WTPs in Granada, Spain, a synthetically generated foam dataset, and images from publicly available datasets to improve generalization. This research offers a practical, scalable, and privacy-aware solution for automatic foam tracking in WTPs. The findings highlight the significant potential of integrating large-scale foundational models into FL systems to solve real-world industrial challenges characterized by distributed and sensitive data.

#### Research Highlights
- **Core Innovation:** This paper proposes a new framework to address these challenges by combining Federated Learning (FL) with the state-of-the-art base model for image segmentation, Segment Anything Model 2 (SAM2).
- **Methodology:** The model was trained and validated using various data collections, including real-world images captured at a WTPs in Granada, Spain, a synthetically generated foam dataset, and images from publicly available datasets to improve generalization.
- **Key Finding:** The findings highlight the significant potential of integrating large-scale foundational models into FL systems to solve real-world industrial challenges characterized by distributed and sensitive data..

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
* **Limits:** However, large amounts of labeled data are required to train standard Machine Learning (ML) models.
* **Signal Tags:** #ai #research

---


### Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning
**Date:** 2025-11-12 | **Arxiv:** [2410.11234](https://hub.bitwiki.org/t/bayes-adaptive-monte-carlo-tree-search-for-offline-model-based-reinforcement-learning/23115)

#### Abstract
Offline RL is a powerful approach for data-driven decision-making and control. Compared to model-free methods, offline model-based RL (MBRL) explicitly learns world models from a static dataset and uses them as surrogate simulators, improving the data efficiency and enabling the learned policy to potentially generalize beyond the dataset support. However, there could be various MDPs that behave identically on the offline dataset and dealing with the uncertainty about the true MDP can be challenging. In this paper, we propose modeling offline MBRL as a Bayes Adaptive Markov Decision Process (BAMDP), which is a principled framework for addressing model uncertainty. We further propose a novel Bayes Adaptive Monte-Carlo planning algorithm capable of solving BAMDPs in continuous state and action spaces with stochastic transitions. This planning process is based on Monte Carlo Tree Search and can be integrated into offline MBRL as a policy improvement operator in policy iteration. Our ``RL + Search" framework follows in the footsteps of superhuman AIs like AlphaZero, improving on current offline MBRL methods by incorporating more computation input. The proposed algorithm significantly outperforms state-of-the-art offline RL methods on twelve D4RL MuJoCo tasks and three target tracking tasks in a challenging, stochastic tokamak control simulator. The codebase is available at: https://github.com/LucasCJYSDL/Offline-RL-Kit.

#### Research Highlights
- **Core Innovation:** The proposed algorithm significantly outperforms state-of-the-art offline RL methods on twelve D4RL MuJoCo tasks and three target tracking tasks in a challenging, stochastic tokamak control simulator.
- **Methodology:** Our ``RL + Search" framework follows in the footsteps of superhuman AIs like AlphaZero, improving on current offline MBRL methods by incorporating more computation input.
- **Key Finding:** The codebase is available at: https://github.com/LucasCJYSDL/Offline-RL-Kit..

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
* **Limits:** However, there could be various MDPs that behave identically on the offline dataset and dealing with the uncertainty about the true MDP can be challenging.
* **Signal Tags:** #ai #research

---


### Data-assimilated model-informed reinforcement learning
**Date:** 2025-11-11 | **Arxiv:** [2506.01755](https://hub.bitwiki.org/t/data-assimilated-model-informed-reinforcement-learning/22896)

#### Abstract
The control of spatio-temporally chaos is challenging because of high dimensionality and unpredictability. Model-free reinforcement learning (RL) discovers optimal control policies by interacting with the system, typically requiring observations of the full physical state. In practice, sensors often provide only partial and noisy measurements (observations) of the system. The objective of this paper is to develop a framework that enables the control of chaotic systems with partial and noisy observability. The proposed method, data-assimilated model-informed reinforcement learning (DA-MIRL), integrates (i) low-order models to approximate high-dimensional dynamics; (ii) sequential data assimilation to correct the model prediction when observations become available; and (iii) an off-policy actor-critic RL algorithm to adaptively learn an optimal control strategy based on the corrected state estimates. We test DA-MIRL on the spatiotemporally chaotic solutions of the Kuramoto-Sivashinsky equation. We estimate the full state of the environment with (i) a physics-based model, here, a coarse-grained model; and (ii) a data-driven model, here, the control-aware echo state network, which is proposed in this paper. We show that DA-MIRL successfully estimates and suppresses the chaotic dynamics of the environment in real time from partial observations and approximate models. This work opens opportunities for the control of partially observable chaotic systems.

#### Research Highlights
- **Core Innovation:** We estimate the full state of the environment with (i) a physics-based model, here, a coarse-grained model; and (ii) a data-driven model, here, the control-aware echo state network, which is proposed in this paper.
- **Methodology:** The objective of this paper is to develop a framework that enables the control of chaotic systems with partial and noisy observability.
- **Key Finding:** We show that DA-MIRL successfully estimates and suppresses the chaotic dynamics of the environment in real time from partial observations and approximate models.

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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments
**Date:** 2025-11-11 | **Arxiv:** [2511.07317](https://hub.bitwiki.org/t/rlve-scaling-up-reinforcement-learning-for-language-models-with-adaptive-verifiable-environments/22804)

#### Abstract
We introduce Reinforcement Learning (RL) with Adaptive Verifiable Environments (RLVE), an approach using verifiable environments that procedurally generate problems and provide algorithmically verifiable rewards, to scale up RL for language models (LMs). RLVE enables each verifiable environment to dynamically adapt its problem difficulty distribution to the policy model's capabilities as training progresses. In contrast, static data distributions often lead to vanishing learning signals when problems are either too easy or too hard for the policy. To implement RLVE, we create RLVE-Gym, a large-scale suite of 400 verifiable environments carefully developed through manual environment engineering. Using RLVE-Gym, we show that environment scaling, i.e., expanding the collection of training environments, consistently improves generalizable reasoning capabilities. RLVE with joint training across all 400 environments in RLVE-Gym yields a 3.37% absolute average improvement across six reasoning benchmarks, starting from one of the strongest 1.5B reasoning LMs. By comparison, continuing this LM's original RL training yields only a 0.49% average absolute gain despite using over 3x more compute. We release our code publicly.

#### Research Highlights
- **Core Innovation:** We introduce Reinforcement Learning (RL) with Adaptive Verifiable Environments (RLVE), an approach using verifiable environments that procedurally generate problems and provide algorithmically verifiable rewards, to scale up RL for language models (LMs).
- **Methodology:** By comparison, continuing this LM's original RL training yields only a 0.49% average absolute gain despite using over 3x more compute.
- **Key Finding:** Using RLVE-Gym, we show that environment scaling, i.e., expanding the collection of training environments, consistently improves generalizable reasoning capabilities.

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


### Dissecting Long-Chain-of-Thought Reasoning Models: An Empirical Study
**Date:** 2025-11-11 | **Arxiv:** [2506.04913](https://hub.bitwiki.org/t/dissecting-long-chain-of-thought-reasoning-models-an-empirical-study/22857)

#### Abstract
Despite recent progress in training long-chain-of-thought reasoning models via scaling reinforcement learning (RL), its underlying training dynamics remain poorly understood, and several counterintuitive behaviors persist. This work focuses on three key aspects: (1) We systematically analyze the roles of positive and negative samples in scaling RL, revealing that positive samples mainly facilitate precise fitting to the training data, whereas negative samples significantly enhance generalization and robustness. Interestingly, while positive samples are essential for convergence in the zero-RL setting, training on negative samples alone suffices to attain strong reasoning performance and even better generalization in cold-start scenarios. (2) We identify substantial data inefficiency in group relative policy optimization, where over half of the samples yield zero advantage. To address this, we explore two strategies, including relative length rewards and offline sample injection, to leverage these data better and enhance reasoning efficiency and capability. (3) We investigate unstable performance across various reasoning models and benchmarks, attributing instability to uncertain problems with ambiguous outcomes, and demonstrate that greedy decoding can distort evaluation by flipping the correctness of responses. Our code is available at: https://github.com/takagi97/Dissect-Long-Reason-Models.

#### Research Highlights
- **Core Innovation:** Despite recent progress in training long-chain-of-thought reasoning models via scaling reinforcement learning (RL), its underlying training dynamics remain poorly understood, and several counterintuitive behaviors persist.
- **Methodology:** Despite recent progress in training long-chain-of-thought reasoning models via scaling reinforcement learning (RL), its underlying training dynamics remain poorly understood, and several counterintuitive behaviors persist.
- **Key Finding:** (3) We investigate unstable performance across various reasoning models and benchmarks, attributing instability to uncertain problems with ambiguous outcomes, and demonstrate that greedy decoding can distort evaluation by flipping the correctness of responses.

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


### CT Radiomics-Based Explainable Machine Learning Model for Accurate Differentiation of Malignant and Benign Endometrial Tumors: A Two-Center Study
**Date:** 2025-11-11 | **Arxiv:** [2506.18106](https://hub.bitwiki.org/t/ct-radiomics-based-explainable-machine-learning-model-for-accurate-differentiation-of-malignant-and-benign-endometrial-tumors-a-two-center-study/22902)

#### Abstract
Aimed to develop and validate a CT radiomics-based explainable machine learning model for precise diagnosing malignancy and benignity specifically in endometrial cancer (EC) patients. A total of 83 EC patients from two centers, including 46 with malignant and 37 with benign conditions, were included, with data split into a training set (n=59) and a testing set (n=24). The regions of interest (ROIs) were manually segmented from pre-surgical CT scans, and 1132 radiomic features were extracted from the pre-surgical CT scans using Pyradiomics. Six explainable machine learning (ML) modeling algorithms were implemented respectively, for determining the optimal radiomics pipeline. The diagnostic performance of the radiomic model was evaluated by using sensitivity, specificity, accuracy, precision, F1 score, AUROC, and AUPRC. To enhance clinical understanding and usability, we separately implemented SHAP analysis and feature mapping visualization, and evaluated the calibration curve and decision curve. By comparing six modeling strategies, the Random Forest model emerged as the optimal choice for diagnosing EC, with a training AUROC of 1.00 and a testing AUROC of 0.96. SHAP identified the most important radiomic features, revealing that all selected features were significantly associated with EC (P < 0.05). Radiomics feature maps also provide a feasible assessment tool for clinical applications. Decision Curve Analysis (DCA) indicated a higher net benefit for our model compared to the "All" and "None" strategies, suggesting its clinical utility in identifying high-risk cases and reducing unnecessary interventions. In conclusion, the CT radiomics-based explainable ML model achieved high diagnostic performance, which could be used as an intelligent auxiliary tool for the diagnosis of endometrial cancer.

#### Research Highlights
- **Core Innovation:** Aimed to develop and validate a CT radiomics-based explainable machine learning model for precise diagnosing malignancy and benignity specifically in endometrial cancer (EC) patients.
- **Methodology:** The diagnostic performance of the radiomic model was evaluated by using sensitivity, specificity, accuracy, precision, F1 score, AUROC, and AUPRC.
- **Key Finding:** In conclusion, the CT radiomics-based explainable ML model achieved high diagnostic performance, which could be used as an intelligent auxiliary tool for the diagnosis of endometrial cancer..

#### Technical Context
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


### GPT, But Backwards: Exactly Inverting Language Model Outputs
**Date:** 2025-11-11 | **Arxiv:** [2507.01693](https://hub.bitwiki.org/t/gpt-but-backwards-exactly-inverting-language-model-outputs/22861)

#### Abstract
The task of reconstructing unknown textual inputs to language models is a fundamental auditing primitive that allows us to assess the model's vulnerability to a range of security issues, including stealing hidden system prompts, detecting backdoors, and leaking private data. Existing inversion works assume access to differing levels of information (e.g. requiring input-output examples, the model parameters, intermediate activations or output logits) but oftentimes fail to fully reconstruct the desired input. In this paper, we present the Sparse One-hot Discrete Adam (SODA) algorithm, a search-based inversion method that can accurately reconstruct the input text, given white-box access to the language model and its output. Our experiments demonstrate for the first time that exact language model inversion is possible on both natural language and random inputs. Indeed, SODA achieves respectively 98% and 79% reconstruction rates on inputs with lengths up to 10 tokens. Furthermore, we show that input length and vocabulary size have a far greater impact on the probability of a successful reconstruction than the size of the language model itself, thus allowing us to scale to models from 33M to 3B parameters.

#### Research Highlights
- **Core Innovation:** The task of reconstructing unknown textual inputs to language models is a fundamental auditing primitive that allows us to assess the model's vulnerability to a range of security issues, including stealing hidden system prompts, detecting backdoors, and leaking private data.
- **Methodology:** See abstract.
- **Key Finding:** Furthermore, we show that input length and vocabulary size have a far greater impact on the probability of a successful reconstruction than the size of the language model itself, thus allowing us to scale to models from 33M to 3B parameters..

#### Technical Context
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


### A Deep Learning Model for Predicting Transformation Legality
**Date:** 2025-11-11 | **Arxiv:** [2511.06120](https://hub.bitwiki.org/t/a-deep-learning-model-for-predicting-transformation-legality/22742)

#### Abstract
Compilers must check the legality of code transformations to guarantee the correctness of applying a sequence of code transformations to a given code. While such a legality check needs to be precisely computed in general, we can use an approximate legality prediction model in certain cases, such as training a reinforcement learning (RL) agent for schedule prediction. In this paper, we propose an approximate method for legality checks. We propose a novel DL model for predicting the legality of transformations. The model takes the code representation and a list of transformations as input and predicts whether applying those transformations to the code is legal. We implement and evaluate the proposed model, demonstrating its effectiveness. Our evaluation shows an F1 score of 0.91 on a test set of randomly generated programs. To further evaluate the model in a practical scenario, we used the model to replace the legality check used during the training of an RL agent designed for automatic code optimization. We demonstrate that such a replacement enables the agent to train on twice as many steps, resulting in faster training and reducing resource usage by approximately 80\% for CPU and 35\% for RAM. The agent trained using this approach maintains comparable performance, with only a 4\% reduction on benchmarks from the Polybench suite compared to the traditional method.

#### Research Highlights
- **Core Innovation:** We implement and evaluate the proposed model, demonstrating its effectiveness.
- **Methodology:** The agent trained using this approach maintains comparable performance, with only a 4\% reduction on benchmarks from the Polybench suite compared to the traditional method..
- **Key Finding:** We demonstrate that such a replacement enables the agent to train on twice as many steps, resulting in faster training and reducing resource usage by approximately 80\% for CPU and 35\% for RAM.

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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### CG-TTRL: Context-Guided Test-Time Reinforcement Learning for On-Device Large Language Models
**Date:** 2025-11-11 | **Arxiv:** [2511.06430](https://hub.bitwiki.org/t/cg-ttrl-context-guided-test-time-reinforcement-learning-for-on-device-large-language-models/22598)

#### Abstract
Test-time Reinforcement Learning (TTRL) has shown promise in adapting foundation models for complex tasks at test-time, resulting in large performance improvements. TTRL leverages an elegant two-phase sampling strategy: first, multi-sampling derives a pseudo-label via majority voting, while subsequent downsampling and reward-based fine-tuning encourages the model to explore and learn diverse valid solutions, with the pseudo-label modulating the reward signal. Meanwhile, in-context learning has been widely explored at inference time and demonstrated the ability to enhance model performance without weight updates. However, TTRL's two-phase sampling strategy under-utilizes contextual guidance, which can potentially improve pseudo-label accuracy in the initial exploitation phase while regulating exploration in the second. To address this, we propose context-guided TTRL (CG-TTRL), integrating context dynamically into both sampling phases and propose a method for efficient context selection for on-device applications. Our evaluations on mathematical and scientific QA benchmarks show CG-TTRL outperforms TTRL (e.g. additional 7% relative accuracy improvement over TTRL), while boosting efficiency by obtaining strong performance after only a few steps of test-time training (e.g. 8% relative improvement rather than 1% over TTRL after 3 steps).

#### Research Highlights
- **Core Innovation:** To address this, we propose context-guided TTRL (CG-TTRL), integrating context dynamically into both sampling phases and propose a method for efficient context selection for on-device applications.
- **Methodology:** TTRL leverages an elegant two-phase sampling strategy: first, multi-sampling derives a pseudo-label via majority voting, while subsequent downsampling and reward-based fine-tuning encourages the model to explore and learn diverse valid solutions, with the pseudo-label modulating the reward signal.
- **Key Finding:** Our evaluations on mathematical and scientific QA benchmarks show CG-TTRL outperforms TTRL (e.g.

#### Technical Context
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
* **Limits:** However, TTRL's two-phase sampling strategy under-utilizes contextual guidance, which can potentially improve pseudo-label accuracy in the initial exploitation phase while regulating exploration in the second.
* **Signal Tags:** #ai #research

---


### Distributional Surgery for Language Model Activations
**Date:** 2025-11-11 | **Arxiv:** [2501.15758](https://hub.bitwiki.org/t/distributional-surgery-for-language-model-activations/22836)

#### Abstract
Language models, while capable of generating remarkably coherent and seemingly accurate text, can occasionally produce undesirable content, including harmful or toxic outputs. In this paper, we present a new two-stage approach to detect and mitigate undesirable content generations by rectifying activations. First, we train an ensemble of layerwise classifiers to detect undesirable content using activations by minimizing a smooth surrogate of the risk-aware score. Then, for detected undesirable contents, we propose layerwise distributional steering policies that transform the attention heads. These policies are computed through principled semidefinite programming, which aims to minimally perturb the attention distribution while probabilistically guaranteeing the effectiveness of the editions. Empirical evaluations across multiple language models and datasets show that our method outperforms baselines in reducing the generation of undesirable output.

#### Research Highlights
- **Core Innovation:** Then, for detected undesirable contents, we propose layerwise distributional steering policies that transform the attention heads.
- **Methodology:** First, we train an ensemble of layerwise classifiers to detect undesirable content using activations by minimizing a smooth surrogate of the risk-aware score.
- **Key Finding:** Empirical evaluations across multiple language models and datasets show that our method outperforms baselines in reducing the generation of undesirable output..

#### Technical Context
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


### Fast Riemannian-manifold Hamiltonian Monte Carlo for hierarchical Gaussian-process models
**Date:** 2025-11-11 | **Arxiv:** [2511.06407](https://hub.bitwiki.org/t/fast-riemannian-manifold-hamiltonian-monte-carlo-for-hierarchical-gaussian-process-models/22438)

#### Abstract
Hierarchical Bayesian models based on Gaussian processes are considered useful for describing complex nonlinear statistical dependencies among variables in real-world data. However, effective Monte Carlo algorithms for inference with these models have not yet been established, except for several simple cases. In this study, we show that, compared with the slow inference achieved with existing program libraries, the performance of Riemannian-manifold Hamiltonian Monte Carlo (RMHMC) can be drastically improved by optimising the computation order according to the model structure and dynamically programming the eigendecomposition. This improvement cannot be achieved when using an existing library based on a naive automatic differentiator. We numerically demonstrate that RMHMC effectively samples from the posterior, allowing the calculation of model evidence, in a Bayesian logistic regression on simulated data and in the estimation of propensity functions for the American national medical expenditure data using several Bayesian multiple-kernel models. These results lay a foundation for implementing effective Monte Carlo algorithms for analysing real-world data with Gaussian processes, and highlight the need to develop a customisable library set that allows users to incorporate dynamically programmed objects and finely optimises the mode of automatic differentiation depending on the model structure.

#### Research Highlights
- **Core Innovation:** Hierarchical Bayesian models based on Gaussian processes are considered useful for describing complex nonlinear statistical dependencies among variables in real-world data.
- **Methodology:** We numerically demonstrate that RMHMC effectively samples from the posterior, allowing the calculation of model evidence, in a Bayesian logistic regression on simulated data and in the estimation of propensity functions for the American national medical expenditure data using several Bayesian multiple-kernel models.
- **Key Finding:** These results lay a foundation for implementing effective Monte Carlo algorithms for analysing real-world data with Gaussian processes, and highlight the need to develop a customisable library set that allows users to incorporate dynamically programmed objects and finely optimises the mode of automatic differentiation depending on the model structure..

#### Technical Context
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
* **Limits:** However, effective Monte Carlo algorithms for inference with these models have not yet been established, except for several simple cases.
* **Signal Tags:** #ai #research

---


### Bridging Theory and Practice: A Stochastic Learning-Optimization Model for Resilient Automotive Supply Chains
**Date:** 2025-11-11 | **Arxiv:** [2511.06479](https://hub.bitwiki.org/t/bridging-theory-and-practice-a-stochastic-learning-optimization-model-for-resilient-automotive-supply-chains/22442)

#### Abstract
Supply chain disruptions and volatile demand pose significant challenges to the UK automotive industry, which relies heavily on Just-In-Time (JIT) manufacturing. While qualitative studies highlight the potential of integrating Artificial Intelligence (AI) with traditional optimization, a formal, quantitative demonstration of this synergy is lacking. This paper introduces a novel stochastic learning-optimization framework that integrates Bayesian inference with inventory optimization for supply chain management (SCM). We model a two-echelon inventory system subject to stochastic demand and supply disruptions, comparing a traditional static optimization policy against an adaptive policy where Bayesian learning continuously updates parameter estimates to inform stochastic optimization. Our simulations over 365 periods across three operational scenarios demonstrate that the integrated approach achieves 7.4\% cost reduction in stable environments and 5.7\% improvement during supply disruptions, while revealing important limitations during sudden demand shocks due to the inherent conservatism of Bayesian updating. This work provides mathematical validation for practitioner observations and establishes a formal framework for understanding AI-driven supply chain resilience, while identifying critical boundary conditions for successful implementation.

#### Research Highlights
- **Core Innovation:** This paper introduces a novel stochastic learning-optimization framework that integrates Bayesian inference with inventory optimization for supply chain management (SCM).
- **Methodology:** This work provides mathematical validation for practitioner observations and establishes a formal framework for understanding AI-driven supply chain resilience, while identifying critical boundary conditions for successful implementation..
- **Key Finding:** Our simulations over 365 periods across three operational scenarios demonstrate that the integrated approach achieves 7.4\% cost reduction in stable environments and 5.7\% improvement during supply disruptions, while revealing important limitations during sudden demand shocks due to the inherent conservatism of Bayesian updating.

#### Technical Context
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
* **Limits:** limitations during sudden demand shocks due to the inherent conservatism of Bayesian updating.
* **Signal Tags:** #ai #research

---


### Guided by Stars: Interpretable Concept Learning Over Time Series via Temporal Logic Semantics
**Date:** 2025-11-07 | **Arxiv:** [2511.04244](https://hub.bitwiki.org/t/guided-by-stars-interpretable-concept-learning-over-time-series-via-temporal-logic-semantics/22057)

#### Abstract
Time series classification is a task of paramount importance, as this kind of data often arises in safety-critical applications. However, it is typically tackled with black-box deep learning methods, making it hard for humans to understand the rationale behind their output. To take on this challenge, we propose a novel approach, STELLE (Signal Temporal logic Embedding for Logically-grounded Learning and Explanation), a neuro-symbolic framework that unifies classification and explanation through direct embedding of trajectories into a space of temporal logic concepts. By introducing a novel STL-inspired kernel that maps raw time series to their alignment with predefined STL formulae, our model jointly optimises accuracy and interpretability, as each prediction is accompanied by the most relevant logical concepts that characterise it. This yields (i) local explanations as human-readable STL conditions justifying individual predictions, and (ii) global explanations as class-characterising formulae. Experiments demonstrate that STELLE achieves competitive accuracy while providing logically faithful explanations, validated on diverse real-world benchmarks.

#### Research Highlights
- **Core Innovation:** To take on this challenge, we propose a novel approach, STELLE (Signal Temporal logic Embedding for Logically-grounded Learning and Explanation), a neuro-symbolic framework that unifies classification and explanation through direct embedding of trajectories into a space of temporal logic concepts.
- **Methodology:** To take on this challenge, we propose a novel approach, STELLE (Signal Temporal logic Embedding for Logically-grounded Learning and Explanation), a neuro-symbolic framework that unifies classification and explanation through direct embedding of trajectories into a space of temporal logic concepts.
- **Key Finding:** Experiments demonstrate that STELLE achieves competitive accuracy while providing logically faithful explanations, validated on diverse real-world benchmarks..

#### Technical Context
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
* **Limits:** However, it is typically tackled with black-box deep learning methods, making it hard for humans to understand the rationale behind their output.
* **Signal Tags:** #ai #research

---


### YOLO-SAT: A Data-based and Model-based Enhanced YOLOv12 Model for Desert Waste Detection and Classification
**Date:** 2025-11-07 | **Arxiv:** [2511.03888](https://hub.bitwiki.org/t/yolo-sat-a-data-based-and-model-based-enhanced-yolov12-model-for-desert-waste-detection-and-classification/22097)

#### Abstract
The global waste crisis is escalating, with solid waste generation expected to increase tremendously in the coming years. Traditional waste collection methods, particularly in remote or harsh environments like deserts, are labor-intensive, inefficient, and often hazardous. Recent advances in computer vision and deep learning have opened the door to automated waste detection systems, yet most research focuses on urban environments and recyclable materials, overlooking organic and hazardous waste and underexplored terrains such as deserts. In this work, we propose YOLO-SAT, an enhanced real-time object detection framework based on a pruned, lightweight version of YOLOv12 integrated with Self-Adversarial Training (SAT) and specialized data augmentation strategies. Using the DroneTrashNet dataset, we demonstrate significant improvements in precision, recall, and mean average precision (mAP), while achieving low latency and compact model size suitable for deployment on resource-constrained aerial drones. Benchmarking YOLO-SAT against state-of-the-art lightweight YOLO variants further highlights its optimal balance of accuracy and efficiency. Our results validate the effectiveness of combining data-centric and model-centric enhancements for robust, real-time waste detection in desert environments.

#### Research Highlights
- **Core Innovation:** In this work, we propose YOLO-SAT, an enhanced real-time object detection framework based on a pruned, lightweight version of YOLOv12 integrated with Self-Adversarial Training (SAT) and specialized data augmentation strategies.
- **Methodology:** Using the DroneTrashNet dataset, we demonstrate significant improvements in precision, recall, and mean average precision (mAP), while achieving low latency and compact model size suitable for deployment on resource-constrained aerial drones.
- **Key Finding:** Our results validate the effectiveness of combining data-centric and model-centric enhancements for robust, real-time waste detection in desert environments..

#### Technical Context
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


### Dynamic causal discovery in Alzheimer's disease through latent pseudotime modelling
**Date:** 2025-11-07 | **Arxiv:** [2511.04619](https://hub.bitwiki.org/t/dynamic-causal-discovery-in-alzheimers-disease-through-latent-pseudotime-modelling/22127)

#### Abstract
The application of causal discovery to diseases like Alzheimer's (AD) is limited by the static graph assumptions of most methods; such models cannot account for an evolving pathophysiology, modulated by a latent disease pseudotime. We propose to apply an existing latent variable model to real-world AD data, inferring a pseudotime that orders patients along a data-driven disease trajectory independent of chronological age, then learning how causal relationships evolve. Pseudotime outperformed age in predicting diagnosis (AUC 0.82 vs 0.59). Incorporating minimal, disease-agnostic background knowledge substantially improved graph accuracy and orientation. Our framework reveals dynamic interactions between novel (NfL, GFAP) and established AD markers, enabling practical causal discovery despite violated assumptions.

#### Research Highlights
- **Core Innovation:** We propose to apply an existing latent variable model to real-world AD data, inferring a pseudotime that orders patients along a data-driven disease trajectory independent of chronological age, then learning how causal relationships evolve.
- **Methodology:** Our framework reveals dynamic interactions between novel (NfL, GFAP) and established AD markers, enabling practical causal discovery despite violated assumptions..
- **Key Finding:** Our framework reveals dynamic interactions between novel (NfL, GFAP) and established AD markers, enabling practical causal discovery despite violated assumptions..

#### Technical Context
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


### Deep Dictionary-Free Method for Identifying Linear Model of Nonlinear System with Input Delay
**Date:** 2025-11-07 | **Arxiv:** [2511.04451](https://hub.bitwiki.org/t/deep-dictionary-free-method-for-identifying-linear-model-of-nonlinear-system-with-input-delay/22115)

#### Abstract
Nonlinear dynamical systems with input delays pose significant challenges for prediction, estimation, and control due to their inherent complexity and the impact of delays on system behavior. Traditional linear control techniques often fail in these contexts, necessitating innovative approaches. This paper introduces a novel approach to approximate the Koopman operator using an LSTM-enhanced Deep Koopman model, enabling linear representations of nonlinear systems with time delays. By incorporating Long Short-Term Memory (LSTM) layers, the proposed framework captures historical dependencies and efficiently encodes time-delayed system dynamics into a latent space. Unlike traditional extended Dynamic Mode Decomposition (eDMD) approaches that rely on predefined dictionaries, the LSTM-enhanced Deep Koopman model is dictionary-free, which mitigates the problems with the underlying dynamics being known and incorporated into the dictionary. Quantitative comparisons with extended eDMD on a simulated system demonstrate highly significant performance gains in prediction accuracy in cases where the true nonlinear dynamics are unknown and achieve comparable results to eDMD with known dynamics of a system.

#### Research Highlights
- **Core Innovation:** By incorporating Long Short-Term Memory (LSTM) layers, the proposed framework captures historical dependencies and efficiently encodes time-delayed system dynamics into a latent space.
- **Methodology:** By incorporating Long Short-Term Memory (LSTM) layers, the proposed framework captures historical dependencies and efficiently encodes time-delayed system dynamics into a latent space.
- **Key Finding:** Quantitative comparisons with extended eDMD on a simulated system demonstrate highly significant performance gains in prediction accuracy in cases where the true nonlinear dynamics are unknown and achieve comparable results to eDMD with known dynamics of a system..

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
* **Limits:** challenges for prediction, estimation, and control due to their inherent complexity and the impact of delays on system behavior.
* **Signal Tags:** #ai #research

---


### Fitting Reinforcement Learning Model to Behavioral Data under Bandits
**Date:** 2025-11-07 | **Arxiv:** [2511.04454](https://hub.bitwiki.org/t/fitting-reinforcement-learning-model-to-behavioral-data-under-bandits/22116)

#### Abstract
We consider the problem of fitting a reinforcement learning (RL) model to some given behavioral data under a multi-armed bandit environment. These models have received much attention in recent years for characterizing human and animal decision making behavior. We provide a generic mathematical optimization problem formulation for the fitting problem of a wide range of RL models that appear frequently in scientific research applications, followed by a detailed theoretical analysis of its convexity properties. Based on the theoretical results, we introduce a novel solution method for the fitting problem of RL models based on convex relaxation and optimization. Our method is then evaluated in several simulated bandit environments to compare with some benchmark methods that appear in the literature. Numerical results indicate that our method achieves comparable performance to the state-of-the-art, while significantly reducing computation time. We also provide an open-source Python package for our proposed method to empower researchers to apply it in the analysis of their datasets directly, without prior knowledge of convex optimization.

#### Research Highlights
- **Core Innovation:** We also provide an open-source Python package for our proposed method to empower researchers to apply it in the analysis of their datasets directly, without prior knowledge of convex optimization..
- **Methodology:** See abstract.
- **Key Finding:** Numerical results indicate that our method achieves comparable performance to the state-of-the-art, while significantly reducing computation time.

#### Technical Context
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


### Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos
**Date:** 2025-11-07 | **Arxiv:** [2506.15680](https://hub.bitwiki.org/t/particle-grid-neural-dynamics-for-learning-deformable-object-models-from-rgb-d-videos/22176)

#### Abstract
Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .

#### Research Highlights
- **Core Innovation:** Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information.
- **Methodology:** Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos.
- **Key Finding:** Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks.

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
* **Limits:** challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation.
* **Signal Tags:** #ai #research

---


### Learning-at-Criticality in Large Language Models for Quantum Field Theory and Beyond
**Date:** 2025-11-07 | **Arxiv:** [2506.03703](https://hub.bitwiki.org/t/learning-at-criticality-in-large-language-models-for-quantum-field-theory-and-beyond/22159)

#### Abstract
Fundamental physics often confronts complex symbolic problems with few guiding exemplars or established principles. While artificial intelligence (AI) offers promise, its typical need for vast datasets to learn from hinders its use in these information-scarce frontiers. We introduce learning at criticality (LaC), a reinforcement learning (RL) scheme that tunes Large Language Models (LLMs) to a sharp learning transition, addressing this information scarcity. At this transition, LLMs achieve peak generalization from minimal data, exemplified by 7-digit base-7 addition -- a test of nontrivial arithmetic reasoning. To elucidate this peak, we analyze a minimal concept-network model (CoNet) designed to capture the essence of how LLMs might link tokens. Trained on a single exemplar, this model also undergoes a sharp learning transition. This transition exhibits hallmarks of a second-order phase transition, notably power-law distributed solution path lengths. At this critical point, the system maximizes a ``critical thinking pattern" crucial for generalization, enabled by the underlying scale-free exploration. This suggests LLMs reach peak performance by operating at criticality, where such explorative dynamics enable the extraction of underlying operational rules. We demonstrate LaC in quantum field theory: an 8B-parameter LLM, tuned to its critical point by LaC using a few exemplars of symbolic Matsubara sums, solves unseen, higher-order problems, significantly outperforming far larger models. LaC thus leverages critical phenomena, a physical principle, to empower AI for complex, data-sparse challenges in fundamental physics.

#### Research Highlights
- **Core Innovation:** We introduce learning at criticality (LaC), a reinforcement learning (RL) scheme that tunes Large Language Models (LLMs) to a sharp learning transition, addressing this information scarcity.
- **Methodology:** We demonstrate LaC in quantum field theory: an 8B-parameter LLM, tuned to its critical point by LaC using a few exemplars of symbolic Matsubara sums, solves unseen, higher-order problems, significantly outperforming far larger models.
- **Key Finding:** We demonstrate LaC in quantum field theory: an 8B-parameter LLM, tuned to its critical point by LaC using a few exemplars of symbolic Matsubara sums, solves unseen, higher-order problems, significantly outperforming far larger models.

#### Technical Context
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
* **Limits:** challenges in fundamental physics.
* **Signal Tags:** #ai #research

---


### FusionDP: Foundation Model-Assisted Differentially Private Learning for Partially Sensitive Features
**Date:** 2025-11-07 | **Arxiv:** [2511.03806](https://hub.bitwiki.org/t/fusiondp-foundation-model-assisted-differentially-private-learning-for-partially-sensitive-features/21989)

#### Abstract
Ensuring the privacy of sensitive training data is crucial in privacy-preserving machine learning. However, in practical scenarios, privacy protection may be required for only a subset of features. For instance, in ICU data, demographic attributes like age and gender pose higher privacy risks due to their re-identification potential, whereas raw lab results are generally less sensitive. Traditional DP-SGD enforces privacy protection on all features in one sample, leading to excessive noise injection and significant utility degradation. We propose FusionDP, a two-step framework that enhances model utility under feature-level differential privacy. First, FusionDP leverages large foundation models to impute sensitive features given non-sensitive features, treating them as external priors that provide high-quality estimates of sensitive attributes without accessing the true values during model training. Second, we introduce a modified DP-SGD algorithm that trains models on both original and imputed features while formally preserving the privacy of the original sensitive features. We evaluate FusionDP on two modalities: a sepsis prediction task on tabular data from PhysioNet and a clinical note classification task from MIMIC-III. By comparing against privacy-preserving baselines, our results show that FusionDP significantly improves model performance while maintaining rigorous feature-level privacy, demonstrating the potential of foundation model-driven imputation to enhance the privacy-utility trade-off for various modalities.

#### Research Highlights
- **Core Innovation:** Second, we introduce a modified DP-SGD algorithm that trains models on both original and imputed features while formally preserving the privacy of the original sensitive features.
- **Methodology:** We propose FusionDP, a two-step framework that enhances model utility under feature-level differential privacy.
- **Key Finding:** By comparing against privacy-preserving baselines, our results show that FusionDP significantly improves model performance while maintaining rigorous feature-level privacy, demonstrating the potential of foundation model-driven imputation to enhance the privacy-utility trade-off for various modalities..

#### Technical Context
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
* **Limits:** However, in practical scenarios, privacy protection may be required for only a subset of features.
* **Signal Tags:** #ai #research

---


### Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations
**Date:** 2025-11-07 | **Arxiv:** [2511.04000](https://hub.bitwiki.org/t/towards-scalable-meta-learning-of-near-optimal-interpretable-models-via-synthetic-model-generations/22010)

#### Abstract
Decision trees are widely used in high-stakes fields like finance and healthcare due to their interpretability. This work introduces an efficient, scalable method for generating synthetic pre-training data to enable meta-learning of decision trees. Our approach samples near-optimal decision trees synthetically, creating large-scale, realistic datasets. Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees. This strategy significantly reduces computational costs, enhances data generation flexibility, and paves the way for scalable and efficient meta-learning of interpretable decision tree models.

#### Research Highlights
- **Core Innovation:** This work introduces an efficient, scalable method for generating synthetic pre-training data to enable meta-learning of decision trees.
- **Methodology:** Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees.
- **Key Finding:** Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees.

#### Technical Context
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


### Scaffolded Language Models with Language Supervision for Mixed-Autonomy: A Survey
**Date:** 2025-11-05 | **Arxiv:** [2410.16392](https://hub.bitwiki.org/t/scaffolded-language-models-with-language-supervision-for-mixed-autonomy-a-survey/21722)

#### Abstract
This survey organizes the intricate literature on the design and optimization of emerging structures around post-trained LMs. We refer to this overarching structure as scaffolded LMs and focus on LMs that are integrated into multi-step processes with tools. We view scaffolded LMs as semi-parametric models wherein we train non-parametric variables, including the prompt, tools, and scaffold's code. In particular, they interpret instructions, use tools, and receive feedback all in language. Recent works use an LM as an optimizer to interpret language supervision and update non-parametric variables according to intricate objectives. In this survey, we refer to this paradigm as training of scaffolded LMs with language supervision. A key feature of non-parametric training is the ability to learn from language. Parametric training excels in learning from demonstration (supervised learning), exploration (reinforcement learning), or observations (unsupervised learning), using well-defined loss functions. Language-based optimization enables rich, interpretable, and expressive objectives, while mitigating issues like catastrophic forgetting and supporting compatibility with closed-source models. Furthermore, agents are increasingly deployed as co-workers in real-world applications such as Copilot in Office tools or software development. In these mixed-autonomy settings, where control and decision-making are shared between human and AI, users point out errors or suggest corrections. Accordingly, we discuss agents that continuously improve by learning from this real-time, language-based feedback and refer to this setting as streaming learning from language supervision.

#### Research Highlights
- **Core Innovation:** This survey organizes the intricate literature on the design and optimization of emerging structures around post-trained LMs.
- **Methodology:** Parametric training excels in learning from demonstration (supervised learning), exploration (reinforcement learning), or observations (unsupervised learning), using well-defined loss functions.
- **Key Finding:** Accordingly, we discuss agents that continuously improve by learning from this real-time, language-based feedback and refer to this setting as streaming learning from language supervision..

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
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Reinforcement learning based data assimilation for unknown state model
**Date:** 2025-11-05 | **Arxiv:** [2511.02286](https://hub.bitwiki.org/t/reinforcement-learning-based-data-assimilation-for-unknown-state-model/21559)

#### Abstract
Data assimilation (DA) has increasingly emerged as a critical tool for state estimation   across a wide range of applications. It is signiffcantly challenging when the governing equations of the underlying dynamics are unknown. To this end, various machine learning approaches have been employed to construct a surrogate state transition   model in a supervised learning framework, which relies on pre-computed training   datasets. However, it is often infeasible to obtain noise-free ground-truth state sequences in practice. To address this challenge, we propose a novel method that integrates reinforcement learning with ensemble-based Bayesian ffltering methods, enabling   the learning of surrogate state transition model for unknown dynamics directly from noisy observations, without using true state trajectories. Speciffcally, we treat the process for computing maximum likelihood estimation of surrogate model parameters   as a sequential decision-making problem, which can be formulated as a discretetime   Markov decision process (MDP). Under this formulation, learning the surrogate transition model is equivalent to ffnding an optimal policy of the MDP, which can be effectively addressed using reinforcement learning techniques. Once the model is trained offfine, state estimation can be performed in the online stage using ffltering methods based on the learned dynamics. The proposed framework accommodates a wide range of observation scenarios, including nonlinear and partially observed measurement   models. A few numerical examples demonstrate that the proposed method achieves superior accuracy and robustness in high-dimensional settings.

#### Research Highlights
- **Core Innovation:** A few numerical examples demonstrate that the proposed method achieves superior accuracy and robustness in high-dimensional settings..
- **Methodology:** The proposed framework accommodates a wide range of observation scenarios, including nonlinear and partially observed measurement   models.
- **Key Finding:** A few numerical examples demonstrate that the proposed method achieves superior accuracy and robustness in high-dimensional settings..

#### Technical Context
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
* **Limits:** However, it is often infeasible to obtain noise-free ground-truth state sequences in practice.
* **Signal Tags:** #ai #research

---


### Fast, Private, and Protected: Safeguarding Data Privacy and Defending Against Model Poisoning Attacks in Federated Learning
**Date:** 2025-11-05 | **Arxiv:** [2511.02797](https://hub.bitwiki.org/t/fast-private-and-protected-safeguarding-data-privacy-and-defending-against-model-poisoning-attacks-in-federated-learning/21623)

#### Abstract
Federated Learning (FL) is a distributed training paradigm wherein participants collaborate to build a global model while ensuring the privacy of the involved data, which remains stored on participant devices. However, proposals aiming to ensure such privacy also make it challenging to protect against potential attackers seeking to compromise the training outcome. In this context, we present Fast, Private, and Protected (FPP), a novel approach that aims to safeguard federated training while enabling secure aggregation to preserve data privacy. This is accomplished by evaluating rounds using participants' assessments and enabling training recovery after an attack. FPP also employs a reputation-based mechanism to mitigate the participation of attackers. We created a dockerized environment to validate the performance of FPP compared to other approaches in the literature (FedAvg, Power-of-Choice, and aggregation via Trimmed Mean and Median). Our experiments demonstrate that FPP achieves a rapid convergence rate and can converge even in the presence of malicious participants performing model poisoning attacks.

#### Research Highlights
- **Core Innovation:** Federated Learning (FL) is a distributed training paradigm wherein participants collaborate to build a global model while ensuring the privacy of the involved data, which remains stored on participant devices.
- **Methodology:** We created a dockerized environment to validate the performance of FPP compared to other approaches in the literature (FedAvg, Power-of-Choice, and aggregation via Trimmed Mean and Median).
- **Key Finding:** Our experiments demonstrate that FPP achieves a rapid convergence rate and can converge even in the presence of malicious participants performing model poisoning attacks..

#### Technical Context
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
* **Limits:** However, proposals aiming to ensure such privacy also make it challenging to protect against potential attackers seeking to compromise the training outcome.
* **Signal Tags:** #ai #research

---


### Identification of Capture Phases in Nanopore Protein Sequencing Data Using a Deep Learning Model
**Date:** 2025-11-04 | **Arxiv:** [2511.01277](https://hub.bitwiki.org/t/identification-of-capture-phases-in-nanopore-protein-sequencing-data-using-a-deep-learning-model/21210)

#### Abstract
Nanopore protein sequencing produces long, noisy ionic current traces in which key molecular phases, such as protein capture and translocation, are embedded. Capture phases mark the successful entry of a protein into the pore and serve as both a checkpoint and a signal that a channel merits further analysis. However, manual identification of capture phases is time-intensive, often requiring several days for expert reviewers to annotate the data due to the need for domain-specific interpretation of complex signal patterns. To address this, a lightweight one-dimensional convolutional neural network (1D CNN) was developed and trained to detect capture phases in down-sampled signal windows. Evaluated against CNN-LSTM (Long Short-Term Memory) hybrids, histogram-based classifiers, and other CNN variants using run-level data splits, our best model, CaptureNet-Deep, achieved an F1 score of 0.94 and precision of 93.39% on held-out test data. The model supports low-latency inference and is integrated into a dashboard for Oxford Nanopore experiments, reducing the total analysis time from several days to under thirty minutes. These results show that efficient, real-time capture detection is possible using simple, interpretable architectures and suggest a broader role for lightweight ML models in sequencing workflows.

#### Research Highlights
- **Core Innovation:** Nanopore protein sequencing produces long, noisy ionic current traces in which key molecular phases, such as protein capture and translocation, are embedded.
- **Methodology:** These results show that efficient, real-time capture detection is possible using simple, interpretable architectures and suggest a broader role for lightweight ML models in sequencing workflows..
- **Key Finding:** These results show that efficient, real-time capture detection is possible using simple, interpretable architectures and suggest a broader role for lightweight ML models in sequencing workflows..

#### Technical Context
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
* **Limits:** However, manual identification of capture phases is time-intensive, often requiring several days for expert reviewers to annotate the data due to the need for domain-specific interpretation of complex signal patterns.
* **Signal Tags:** #ai #research

---


### Priors in Time: Missing Inductive Biases for Language Model Interpretability
**Date:** 2025-11-04 | **Arxiv:** [2511.01836](https://hub.bitwiki.org/t/priors-in-time-missing-inductive-biases-for-language-model-interpretability/21246)

#### Abstract
Recovering meaningful concepts from language model activations is a central aim of interpretability. While existing feature extraction methods aim to identify concepts that are independent directions, it is unclear if this assumption can capture the rich temporal structure of language. Specifically, via a Bayesian lens, we demonstrate that Sparse Autoencoders (SAEs) impose priors that assume independence of concepts across time, implying stationarity. Meanwhile, language model representations exhibit rich temporal dynamics, including systematic growth in conceptual dimensionality, context-dependent correlations, and pronounced non-stationarity, in direct conflict with the priors of SAEs. Taking inspiration from computational neuroscience, we introduce a new interpretability objective -- Temporal Feature Analysis -- which possesses a temporal inductive bias to decompose representations at a given time into two parts: a predictable component, which can be inferred from the context, and a residual component, which captures novel information unexplained by the context. Temporal Feature Analyzers correctly parse garden path sentences, identify event boundaries, and more broadly delineate abstract, slow-moving information from novel, fast-moving information, while existing SAEs show significant pitfalls in all the above tasks. Overall, our results underscore the need for inductive biases that match the data in designing robust interpretability tools.

#### Research Highlights
- **Core Innovation:** Taking inspiration from computational neuroscience, we introduce a new interpretability objective -- Temporal Feature Analysis -- which possesses a temporal inductive bias to decompose representations at a given time into two parts: a predictable component, which can be inferred from the context, and a residual component, which captures novel information unexplained by the context.
- **Methodology:** Specifically, via a Bayesian lens, we demonstrate that Sparse Autoencoders (SAEs) impose priors that assume independence of concepts across time, implying stationarity.
- **Key Finding:** Overall, our results underscore the need for inductive biases that match the data in designing robust interpretability tools..

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


### Collaborative Large Language Model Inference via Resource-Aware Parallel Speculative Decoding
**Date:** 2025-11-04 | **Arxiv:** [2511.01695](https://hub.bitwiki.org/t/collaborative-large-language-model-inference-via-resource-aware-parallel-speculative-decoding/21232)

#### Abstract
The growing demand for on-device large language model (LLM) inference highlights the need for efficient mobile edge computing (MEC) solutions, especially in resource-constrained settings. Speculative decoding offers a promising solution by partitioning token generation between a lightweight draft model on mobile devices and a powerful target model on edge servers, but suffers from communication overhead and asynchronous delays. This paper is the first to propose a unified framework that jointly optimizes user association and resource allocation (UARA) to support efficient parallel speculative decoding. We solve the UARA problem using a multi-agent deep reinforcement learning algorithm. To evaluate our approach under realistic conditions, we conduct experiments using the Sionna simulator. Results show that our method achieves up to 28.0% and an average of 23.7% reduction in end-to-end latency without compromising inference accuracy, enabling scalable and low-latency LLM services in MEC systems.

#### Research Highlights
- **Core Innovation:** This paper is the first to propose a unified framework that jointly optimizes user association and resource allocation (UARA) to support efficient parallel speculative decoding.
- **Methodology:** To evaluate our approach under realistic conditions, we conduct experiments using the Sionna simulator.
- **Key Finding:** Results show that our method achieves up to 28.0% and an average of 23.7% reduction in end-to-end latency without compromising inference accuracy, enabling scalable and low-latency LLM services in MEC systems..

#### Technical Context
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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated in abstract.
* **Signal Tags:** #ai #research

---


### Chitchat with AI: Understand the supply chain carbon disclosure of companies worldwide through Large Language Model
**Date:** 2025-11-04 | **Arxiv:** [2511.00024](https://hub.bitwiki.org/t/chitchat-with-ai-understand-the-supply-chain-carbon-disclosure-of-companies-worldwide-through-large-language-model/21252)

#### Abstract
In the context of global sustainability mandates, corporate carbon disclosure has emerged as a critical mechanism for aligning business strategy with environmental responsibility. The Carbon Disclosure Project (CDP) hosts the world's largest longitudinal dataset of climate-related survey responses, combining structured indicators with open-ended narratives, but the heterogeneity and free-form nature of these disclosures present significant analytical challenges for benchmarking, compliance monitoring, and investment screening. This paper proposes a novel decision-support framework that leverages large language models (LLMs) to assess corporate climate disclosure quality at scale. It develops a master rubric that harmonizes narrative scoring across 11 years of CDP data (2010-2020), enabling cross-sector and cross-country benchmarking. By integrating rubric-guided scoring with percentile-based normalization, our method identifies temporal trends, strategic alignment patterns, and inconsistencies in disclosure across industries and regions. Results reveal that sectors such as technology and countries like Germany consistently demonstrate higher rubric alignment, while others exhibit volatility or superficial engagement, offering insights that inform key decision-making processes for investors, regulators, and corporate environmental, social, and governance (ESG) strategists. The proposed LLM-based approach transforms unstructured disclosures into quantifiable, interpretable, comparable, and actionable intelligence, advancing the capabilities of AI-enabled decision support systems (DSSs) in the domain of climate governance.

#### Research Highlights
- **Core Innovation:** The proposed LLM-based approach transforms unstructured disclosures into quantifiable, interpretable, comparable, and actionable intelligence, advancing the capabilities of AI-enabled decision support systems (DSSs) in the domain of climate governance..
- **Methodology:** This paper proposes a novel decision-support framework that leverages large language models (LLMs) to assess corporate climate disclosure quality at scale.
- **Key Finding:** Results reveal that sectors such as technology and countries like Germany consistently demonstrate higher rubric alignment, while others exhibit volatility or superficial engagement, offering insights that inform key decision-making processes for investors, regulators, and corporate environmental, social, and governance (ESG) strategists.

#### Technical Context
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
* **Limits:** challenges for benchmarking, compliance monitoring, and investment screening.
* **Signal Tags:** #ai #research

---
