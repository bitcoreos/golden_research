# Vol 10 Causal Inference   Logic
*Enriched by BITCOREOS | Phase 4 Batch 2*

---

### Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
**Date:** 2025-11-14 | **Arxiv:** [2506.01939](https://arxiv.org/abs/2506.01939)

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### From Invariant Representations to Invariant Data: Provable Robustness to Spurious Correlations via Noisy Counterfactual Matching
**Date:** 2025-11-11 | **Arxiv:** [2505.24843](https://arxiv.org/abs/2505.24843)

#### Abstract
Models that learn spurious correlations from training data often fail when deployed in new environments. While many methods aim to learn invariant representations to address this, they often underperform standard empirical risk minimization (ERM). We propose a data-centric alternative that shifts the focus from learning invariant representations to leveraging invariant data pairs -- pairs of samples that should have the same prediction. We prove that certain counterfactuals naturally satisfy this invariance property. Based on this, we introduce Noisy Counterfactual Matching (NCM), a simple constraint-based method that improves robustness by leveraging even a small number of \emph{noisy} counterfactual pairs -- improving upon prior works that do not explicitly consider noise. For linear causal models, we prove that NCM's test-domain error is bounded by its in-domain error plus a term dependent on the counterfactuals' quality and diversity. Experiments on synthetic data validate our theory, and we demonstrate NCM's effectiveness on real-world datasets.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards
**Date:** 2025-10-21 | **Arxiv:** [2505.24760](https://arxiv.org/abs/2505.24760)

#### Abstract
We introduce Reasoning Gym (RG), a library of reasoning environments for reinforcement learning with verifiable rewards. It provides over 100 data generators and verifiers spanning multiple domains including algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and various common games. Its key innovation is the ability to generate virtually infinite training data with adjustable complexity, unlike most previous reasoning datasets, which are typically fixed. This procedural generation approach allows for continuous evaluation across varying difficulty levels. Our experimental results demonstrate the efficacy of RG in both evaluating and reinforcement learning of reasoning models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Think Just Enough: Sequence-Level Entropy as a Confidence Signal for LLM Reasoning
**Date:** 2025-10-10 | **Arxiv:** [2510.08146](https://arxiv.org/abs/2510.08146)

#### Abstract
We introduce a simple, yet novel entropy-based framework to drive token efficiency in large language models during reasoning tasks. Our approach uses Shannon entropy from token-level logprobs as a confidence signal to enable early stopping, achieving 25-50% computational savings while maintaining task accuracy. Crucially, we demonstrate that entropy-based confidence calibration represents an emergent property of advanced post-training optimization present in modern reasoning models but notably absent in standard instruction-tuned and pre-trained models (Llama 3.3 70B). We show that the entropy threshold to stop reasoning varies from model to model but can be calculated easily in one shot using only a few examples from existing reasoning datasets. Our results indicate that advanced reasoning models often know that they've gotten a correct answer early on, and that this emergent confidence awareness can be exploited to save tokens and reduce latency. The framework demonstrates consistent performance across reasoning-optimized model families with 25-50% computational cost reduction while preserving accuracy, revealing that confidence mechanisms represent a distinguishing characteristic of modern post-trained reasoning systems versus their predecessors.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Third Pillar of Causal Analysis? A Measurement Perspective on Causal Representations
**Date:** 2025-11-18 | **Arxiv:** [2505.17708](https://arxiv.org/abs/2505.17708)

#### Abstract
Causal reasoning and discovery, two fundamental tasks of causal analysis, often face challenges in applications due to the complexity, noisiness, and high-dimensionality of real-world data. Despite recent progress in identifying latent causal structures using causal representation learning (CRL), what makes learned representations useful for causal downstream tasks and how to evaluate them are still not well understood. In this paper, we reinterpret CRL using a measurement model framework, where the learned representations are viewed as proxy measurements of the latent causal variables. Our approach clarifies the conditions under which learned representations support downstream causal reasoning and provides a principled basis for quantitatively assessing the quality of representations using a new Test-based Measurement EXclusivity (T-MEX) score. We validate T-MEX across diverse causal inference scenarios, including numerical simulations and real-world ecological video analysis, demonstrating that the proposed framework and corresponding score effectively assess the identification of learned representations and their usefulness for causal downstream tasks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Latent Planning via Embedding Arithmetic: A Contrastive Approach to Strategic Reasoning
**Date:** 2025-11-13 | **Arxiv:** [2511.09477](https://arxiv.org/abs/2511.09477)

#### Abstract
Planning in high-dimensional decision spaces is increasingly being studied through the lens of learned representations. Rather than training policies or value heads, we investigate whether planning can be carried out directly in an evaluation-aligned embedding space. We introduce SOLIS, which learns such a space using supervised contrastive learning. In this representation, outcome similarity is captured by proximity, and a single global advantage vector orients the space from losing to winning regions. Candidate actions are then ranked according to their alignment with this direction, reducing planning to vector operations in latent space. We demonstrate this approach in chess, where SOLIS uses only a shallow search guided by the learned embedding to reach competitive strength under constrained conditions. More broadly, our results suggest that evaluation-aligned latent planning offers a lightweight alternative to traditional dynamics models or policy learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
**Date:** 2025-10-27 | **Arxiv:** [2503.05179](https://arxiv.org/abs/2503.05179)

#### Abstract
Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 18 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 84% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Reasoning's Razor: Reasoning Improves Accuracy but Can Hurt Recall at Critical Operating Points in Safety and Hallucination Detection
**Date:** 2025-10-27 | **Arxiv:** [2510.21049](https://arxiv.org/abs/2510.21049)

#### Abstract
Reasoning has become a central paradigm for large language models (LLMs), consistently boosting accuracy across diverse benchmarks. Yet its suitability for precision-sensitive tasks remains unclear. We present the first systematic study of reasoning for classification tasks under strict low false positive rate (FPR) regimes. Our analysis covers two tasks--safety detection and hallucination detection--evaluated in both fine-tuned and zero-shot settings, using standard LLMs and Large Reasoning Models (LRMs). Our results reveal a clear trade-off: Think On (reasoning-augmented) generation improves overall accuracy, but underperforms at the low-FPR thresholds essential for practical use. In contrast, Think Off (no reasoning during inference) dominates in these precision-sensitive regimes, with Think On surpassing only when higher FPRs are acceptable. In addition, we find token-based scoring substantially outperforms self-verbalized confidence for precision-sensitive deployments. Finally, a simple ensemble of the two modes recovers the strengths of each. Taken together, our findings position reasoning as a double-edged tool: beneficial for average accuracy, but often ill-suited for applications requiring strict precision.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Local Causal Discovery for Statistically Efficient Causal Inference
**Date:** 2025-10-17 | **Arxiv:** [2510.14582](https://arxiv.org/abs/2510.14582)

#### Abstract
Causal discovery methods can identify valid adjustment sets for causal effect estimation for a pair of target variables, even when the underlying causal graph is unknown. Global causal discovery methods focus on learning the whole causal graph and therefore enable the recovery of optimal adjustment sets, i.e., sets with the lowest asymptotic variance, but they quickly become computationally prohibitive as the number of variables grows. Local causal discovery methods offer a more scalable alternative by focusing on the local neighborhood of the target variables, but are restricted to statistically suboptimal adjustment sets. In this work, we propose Local Optimal Adjustments Discovery (LOAD), a sound and complete causal discovery approach that combines the computational efficiency of local methods with the statistical optimality of global methods. First, LOAD identifies the causal relation between the targets and tests if the causal effect is identifiable by using only local information. If it is identifiable, it then finds the optimal adjustment set by leveraging local causal discovery to infer the mediators and their parents. Otherwise, it returns the locally valid parent adjustment sets based on the learned local structure. In our experiments on synthetic and realistic data LOAD outperforms global methods in scalability, while providing more accurate effect estimation than local methods.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Improved Monte Carlo Planning via Causal Disentanglement for Structurally-Decomposed Markov Decision Processes
**Date:** 2025-10-06 | **Arxiv:** [2406.16151](https://arxiv.org/abs/2406.16151)

#### Abstract
Markov Decision Processes (MDPs), as a general-purpose framework, often overlook the benefits of incorporating the causal structure of the transition and reward dynamics. For a subclass of resource allocation problems, we introduce the Structurally Decomposed MDP (SD-MDP), which leverages causal disentanglement to partition an MDP's temporal causal graph into independent components. By exploiting this disentanglement, SD-MDP enables dimensionality reduction and computational efficiency gains in optimal value function estimation. We reduce the sequential optimization problem to a fractional knapsack problem with log-linear complexity $O(T \log T)$, outperforming traditional stochastic programming methods that exhibit polynomial complexity with respect to the time horizon $T$. Additionally, SD-MDP's computational advantages are independent of state-action space size, making it viable for high-dimensional spaces. Furthermore, our approach integrates seamlessly with Monte Carlo Tree Search (MCTS), achieving higher expected rewards under constrained simulation budgets while providing a vanishing simple regret bound. Empirical results demonstrate superior policy performance over benchmarks across various logistics and finance domains.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### PRISM-Physics: Causal DAG-Based Process Evaluation for Physics Reasoning
**Date:** 2025-10-06 | **Arxiv:** [2510.03185](https://arxiv.org/abs/2510.03185)

#### Abstract
Benchmarks for competition-style reasoning have advanced evaluation in mathematics and programming, yet physics remains comparatively explored. Most existing physics benchmarks evaluate only final answers, which fail to capture reasoning processes, while recent stepwise methods rely on heuristic LLM-as-judge scoring or restrictive linear assumptions, limiting reliability and diagnostic validity. We introduce PRISM-Physics, a process-level evaluation framework and benchmark for complex physics reasoning problems. Solutions are represented as directed acyclic graphs (DAGs) of formulas, explicitly encoding causal dependencies among intermediate steps to enable fine-grained, interpretable, and theoretically grounded scoring. We prove the optimality of the DAG representation and the corresponding scoring policy. Combining with a fully rule-based method for symbolic formula equivalence matching that we developed, we ensure consistent validation across diverse formulations without heuristic judgments. Results show that our evaluation framework is more aligned with human experts' scoring. Experiments on state-of-the-art LLMs reveal persistent reasoning failures in physics, while step-level scoring offers both diagnostic insight and rich signals for later training. By combining structural rigor, theoretical guarantees, and symbolic validation, PRISM-Physics provides a principled foundation for advancing process-level evaluation and guiding the development of models with deeper scientific reasoning capabilities.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning
**Date:** 2025-12-08 | **Arxiv:** [2512.05325](https://arxiv.org/abs/2512.05325)

#### Abstract
Large reasoning models achieve strong performance on complex tasks by generating extended chains of thought, but they often "overthink": continuing to reason long after they have enough information to answer correctly. This wastes inference-time compute and can hurt accuracy. Existing attempts to stop early either manipulate decoding with extra sampling and heuristics, rely on auxiliary verifier models, or operate only as post-hoc analysis pipelines without formal guarantees. We introduce LYNX, an online early-exit mechanism that turns a model's own hidden-state awareness into confidence-controlled stopping decisions. LYNX attaches exit decisions to naturally occurring reasoning cues (e.g., "hmm", "wait") during generation, trains a lightweight probe on hidden states at those cue tokens using supervision from forced exits, and wraps the resulting scores in split conformal prediction to obtain distribution-free control over premature exits. Crucially, we train and calibrate this probe once on a generic mathematical corpus and reuse it unchanged across benchmarks, decoding temperatures, and even non-mathematical tasks. Across three model families spanning 1.5B to 32B parameters, a single mathematically trained probe per base model yields strong accuracy--efficiency tradeoffs. On GSM8K, LYNX matches or improves baseline accuracy while reducing tokens by 40--65\%; on MATH-500 it improves accuracy by up to 12 points with roughly 35--60\% fewer tokens; on AIME 2024 it recovers baseline accuracy with more than 50\% token savings; and on CommonsenseQA, a non-math benchmark, it transfers zero-shot with modest accuracy gains and up to 70\% fewer tokens. Compared to state-of-the-art early-exit methods, LYNX offers competitive or superior Pareto frontiers while remaining fully online, requiring no proxy models at inference, and providing explicit, user-tunable confidence guarantees.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Semantic Soft Bootstrapping: Long Context Reasoning in LLMs without Reinforcement Learning
**Date:** 2025-12-05 | **Arxiv:** [2512.05105](https://arxiv.org/abs/2512.05105)

#### Abstract
Long context reasoning in large language models (LLMs) has demonstrated enhancement of their cognitive capabilities via chain-of-thought (CoT) inference. Training such models is usually done via reinforcement learning with verifiable rewards (RLVR) in reasoning based problems, like math and programming. However, RLVR is limited by several bottlenecks, such as, lack of dense reward, and inadequate sample efficiency. As a result, it requires significant compute resources in post-training phase. To overcome these limitations, in this work, we propose \textbf{Semantic Soft Bootstrapping (SSB)}, a self-distillation technique, in which the same base language model plays the role of both teacher and student, but receives different semantic contexts about the correctness of its outcome at training time. The model is first prompted with a math problem and several rollouts are generated. From them, the correct and most common incorrect response are filtered, and then provided to the model in context to produce a more robust, step-by-step explanation with a verified final answer. This pipeline automatically curates a paired teacher-student training set from raw problem-answer data, without any human intervention. This generation process also produces a sequence of logits, which is what the student model tries to match in the training phase just from the bare question alone. In our experiment, Qwen2.5-3B-Instruct on GSM8K dataset via parameter-efficient fine-tuning. We then tested its accuracy on MATH500, and AIME2024 benchmarks. Our experiments show a jump of 10.6%, and 10% improvements in accuracy, respectively, over group relative policy optimization (GRPO), which is a commonly used RLVR algorithm. Our code is available at https://github.com/purbeshmitra/semantic-soft-bootstrapping, and the model, curated dataset is available at https://huggingface.co/purbeshmitra/semantic-soft-bootstrapping.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, RLVR is limited by several bottlenecks, such as, lack of dense reward, and inadequate sample efficiency.
* **Signal Tags:** #ai

---


### A New Causal Rule Learning Approach to Interpretable Estimation of Heterogeneous Treatment Effect
**Date:** 2025-11-24 | **Arxiv:** [2310.06746](https://arxiv.org/abs/2310.06746)

#### Abstract
Interpretability plays a crucial role in the application of statistical learning to estimate heterogeneous treatment effects (HTE) in complex diseases. In this study, we leverage a rule-based workflow, namely causal rule learning (CRL), to estimate and improve our understanding of HTE for atrial septal defect, addressing an overlooked question in the previous literature: what if an individual simultaneously belongs to multiple groups with different average treatment effects? The CRL process consists of three steps: rule discovery, which generates a set of causal rules with corresponding subgroup average treatment effects; rule selection, which identifies a subset of these rules to deconstruct individual-level treatment effects as a linear combination of subgroup-level effects; and rule analysis, which presents a detailed procedure for further analyzing each selected rule from multiple perspectives to identify the most promising rules for validation. Extensive simulation studies and real-world data analysis demonstrate that CRL outperforms other methods in providing interpretable estimates of HTE, especially when dealing with complex ground truth and sufficient sample sizes.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data: "One Map, Many Trials" in Satellite-Driven Poverty Analysis
**Date:** 2025-11-14 | **Arxiv:** [2508.01341](https://arxiv.org/abs/2508.01341)

#### Abstract
Machine learning models trained on Earth observation data, such as satellite imagery, have demonstrated significant promise in predicting household-level wealth indices, enabling the creation of high-resolution wealth maps that can be leveraged across multiple causal trials while addressing chronic data scarcity in global development research. However, because standard training objectives prioritize overall predictive accuracy, these predictions often suffer from shrinkage toward the mean, leading to attenuated estimates of causal treatment effects and limiting their utility in policy evaluations. Existing debiasing methods, such as Prediction-Powered Inference (PPI), can handle this attenuation bias but require additional fresh ground-truth data at the downstream stage of causal inference, which restricts their applicability in data-scarce environments. We introduce and evaluate two post-hoc correction methods -- Linear Calibration Correction (LCC) and a Tweedie's correction approach -- that substantially reduce shrinkage-induced prediction bias without relying on newly collected labeled data. LCC applies a simple linear transformation estimated on a held-out calibration split; Tweedie's method locally de-shrink predictions using density score estimates and a noise scale learned upstream. We provide practical diagnostics for when a correction is warranted and discuss practical limitations. Across analytical results, simulations, and experiments with Demographic and Health Surveys (DHS) data, both approaches reduce attenuation; Tweedie's correction yields nearly unbiased treatment-effect estimates, enabling a "one map, many trials" paradigm. Although we demonstrate on EO-ML wealth mapping, the methods are not geospatial-specific: they apply to any setting where imputed outcomes are reused downstream (e.g., pollution indices, population density, or LLM-derived indicators).

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, because standard training objectives prioritize overall predictive accuracy, these predictions often suffer from shrinkage toward the mean, leading to attenuated estimates of causal treatment effects and limiting their utility in policy evaluations.
* **Signal Tags:** #ai

---


### Semiparametric Double Reinforcement Learning with Applications to Long-Term Causal Inference
**Date:** 2025-11-14 | **Arxiv:** [2501.06926](https://arxiv.org/abs/2501.06926)

#### Abstract
Double Reinforcement Learning (DRL) enables efficient inference for policy values in nonparametric Markov decision processes (MDPs), but existing methods face two major obstacles: (1) they require stringent intertemporal overlap conditions on state trajectories, and (2) they rely on estimating high-dimensional occupancy density ratios. Motivated by problems in long-term causal inference, we extend DRL to a semiparametric setting and develop doubly robust, automatic estimators for general linear functionals of the Q-function in infinite-horizon, time-homogeneous MDPs. By imposing structure on the Q-function, we relax the overlap conditions required by nonparametric methods and obtain efficiency gains. The second obstacle--density-ratio estimation--typically requires computationally expensive and unstable min-max optimization. To address both challenges, we introduce superefficient nonparametric estimators whose limiting variance falls below the generalized Cramer-Rao bound. These estimators treat the Q-function as a one-dimensional summary of the state-action process, reducing high-dimensional overlap requirements to a single-dimensional condition. The procedure is simple to implement: estimate and calibrate the Q-function using fitted Q-iteration, then plug the result into the target functional, thereby avoiding density-ratio estimation altogether.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### FedSDWC: Federated Synergistic Dual-Representation Weak Causal Learning for OOD
**Date:** 2025-11-13 | **Arxiv:** [2511.09036](https://arxiv.org/abs/2511.09036)

#### Abstract
Amid growing demands for data privacy and advances in computational infrastructure, federated learning (FL) has emerged as a prominent distributed learning paradigm. Nevertheless, differences in data distribution (such as covariate and semantic shifts) severely affect its reliability in real-world deployments. To address this issue, we propose FedSDWC, a causal inference method that integrates both invariant and variant features. FedSDWC infers causal semantic representations by modeling the weak causal influence between invariant and variant features, effectively overcoming the limitations of existing invariant learning methods in accurately capturing invariant features and directly constructing causal representations. This approach significantly enhances FL's ability to generalize and detect OOD data. Theoretically, we derive FedSDWC's generalization error bound under specific conditions and, for the first time, establish its relationship with client prior distributions. Moreover, extensive experiments conducted on multiple benchmark datasets validate the superior performance of FedSDWC in handling covariate and semantic shifts. For example, FedSDWC outperforms FedICON, the next best baseline, by an average of 3.04% on CIFAR-10 and 8.11% on CIFAR-100.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### CaberNet: Causal Representation Learning for Cross-Domain HVAC Energy Prediction
**Date:** 2025-11-11 | **Arxiv:** [2511.06634](https://arxiv.org/abs/2511.06634)

#### Abstract
Cross-domain HVAC energy prediction is essential for scalable building energy management, particularly because collecting extensive labeled data for every new building is both costly and impractical. Yet, this task remains highly challenging due to the scarcity and heterogeneity of data across different buildings, climate zones, and seasonal patterns. In particular, buildings situated in distinct climatic regions introduce variability that often leads existing methods to overfit to spurious correlations, rely heavily on expert intervention, or compromise on data diversity. To address these limitations, we propose CaberNet, a causal and interpretable deep sequence model that learns invariant (Markov blanket) representations for robust cross-domain prediction. In a purely data-driven fashion and without requiring any prior knowledge, CaberNet integrates i) a global feature gate trained with a self-supervised Bernoulli regularization to distinguish superior causal features from inferior ones, and ii) a domain-wise training scheme that balances domain contributions, minimizes cross-domain loss variance, and promotes latent factor independence. We evaluate CaberNet on real-world datasets collected from three buildings located in three climatically diverse cities, and it consistently outperforms all baselines, achieving a 22.9% reduction in normalized mean squared error (NMSE) compared to the best benchmark. Our code is available at https://github.com/SusCom-Lab/CaberNet-CRL.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Multi-Task Learning for Visually Grounded Reasoning in Gastrointestinal VQA
**Date:** 2025-11-07 | **Arxiv:** [2511.04384](https://arxiv.org/abs/2511.04384)

#### Abstract
We present a multi-task framework for the MediaEval Medico 2025 challenge, leveraging a LoRA-tuned Florence-2 model for simultaneous visual question answering (VQA), explanation generation, and visual grounding. The proposed system integrates three curated datasets: (1) Kvasir-VQA-x1 for question-answer learning, (2) a synthetically enriched explanation dataset offering structured medical reasoning, and (3) text-to-region pairs linking visual features with segmentation masks. This multi-task setup enables the model to jointly learn visual grounding, reasoning, and interpretation, producing responses that are both accurate and interpretable. Extensive evaluation demonstrates that our approach substantially improves over single-task baselines in both answer accuracy and visual localization, highlighting the effectiveness of grounded multi-task learning for medical VQA applications.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### A Unified Theory for Causal Inference: Direct Debiased Machine Learning via Bregman-Riesz Regression
**Date:** 2025-10-31 | **Arxiv:** [2510.26783](https://arxiv.org/abs/2510.26783)

#### Abstract
This note introduces a unified theory for causal inference that integrates Riesz regression, covariate balancing, density-ratio estimation (DRE), targeted maximum likelihood estimation (TMLE), and the matching estimator in average treatment effect (ATE) estimation. In ATE estimation, the balancing weights and the regression functions of the outcome play important roles, where the balancing weights are referred to as the Riesz representer, bias-correction term, and clever covariates, depending on the context. Riesz regression, covariate balancing, DRE, and the matching estimator are methods for estimating the balancing weights, where Riesz regression is essentially equivalent to DRE in the ATE context, the matching estimator is a special case of DRE, and DRE is in a dual relationship with covariate balancing. TMLE is a method for constructing regression function estimators such that the leading bias term becomes zero. Nearest Neighbor Matching is equivalent to Least Squares Density Ratio Estimation and Riesz Regression.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding
**Date:** 2025-10-29 | **Arxiv:** [2502.05295](https://arxiv.org/abs/2502.05295)

#### Abstract
Estimating causal effects from spatiotemporal observational data is essential in public health, environmental science, and policy evaluation, where randomized experiments are often infeasible. Existing approaches, however, either rely on strong structural assumptions or fail to handle key challenges such as interference, spatial confounding, temporal carryover, and time-varying confounding -- where covariates are influenced by past treatments and, in turn, affect future ones. We introduce GST-UNet (G-computation Spatio-Temporal UNet), a theoretically grounded neural framework that combines a U-Net-based spatiotemporal encoder with regression-based iterative G-computation to estimate location-specific potential outcomes under complex intervention sequences. GST-UNet explicitly adjusts for time-varying confounders and captures non-linear spatial and temporal dependencies, enabling valid causal inference from a single observed trajectory in data-scarce settings. We validate its effectiveness in synthetic experiments and in a real-world analysis of wildfire smoke exposure and respiratory hospitalizations during the 2018 California Camp Fire. Together, these results position GST-UNet as a principled and ready-to-use framework for spatiotemporal causal inference, advancing reliable estimation in policy-relevant and scientific domains.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** however, either rely on strong structural assumptions or fail to handle key challenges such as interference, spatial confounding, temporal carryover, and time-varying confounding -- where covariates are influenced by past treatments and, in turn, affect future ones.
* **Signal Tags:** #ai

---


### FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning
**Date:** 2025-10-24 | **Arxiv:** [2510.19893](https://arxiv.org/abs/2510.19893)

#### Abstract
Medical artificial intelligence systems have achieved remarkable diagnostic capabilities, yet they consistently exhibit performance disparities across demographic groups, causing real-world harm to underrepresented populations. While recent multimodal reasoning foundation models have advanced clinical diagnosis through integrated analysis of diverse medical data, reasoning trainings via reinforcement learning inherit and often amplify biases present in training datasets dominated by majority populations. We introduce Fairness-aware Group Relative Policy Optimization (FairGRPO), a hierarchical reinforcement learning approach that promotes equitable learning across heterogeneous clinical populations. FairGRPO employs adaptive importance weighting of advantages based on representation, task difficulty, and data source. To address the common issue of missing demographic labels in the clinical domain, we further employ unsupervised clustering, which automatically discovers latent demographic groups when labels are unavailable. Through comprehensive experiments across 7 clinical diagnostic datasets spanning 5 clinical modalities across X-ray, CT scan, dermoscropy, mammography and ultrasound, we demonstrate that FairGRPO reduces predictive parity by 27.2% against all vanilla and bias mitigated RL baselines, while improving F1 score by 12.49%. Furthermore, training dynamics analysis reveals that FairGRPO progressively improves fairness throughout optimization, while baseline RL methods exhibit deteriorating fairness as training progresses. Based on FairGRPO, we release FairMedGemma-4B, a fairness-aware clinical VLLM that achieves state-of-the-art performance while demonstrating significantly reduced disparities across demographic groups.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms
**Date:** 2025-10-24 | **Arxiv:** [2505.17190](https://arxiv.org/abs/2505.17190)

#### Abstract
Can algebraic geometry enhance the sharpness, robustness, and interpretability of modern neural reasoning models by equipping them with a mathematically grounded inductive bias? To answer this, we introduce Tropical Attention, an attention mechanism grounded in tropical geometry that lifts the attention kernel into tropical projective space, where reasoning is piecewise-linear and 1-Lipschitz, thus preserving the polyhedral decision structure inherent to combinatorial reasoning. We prove that Multi-Head Tropical Attention (MHTA) stacks universally approximate tropical circuits and realize tropical transitive closure through composition, achieving polynomial resource bounds without invoking recurrent mechanisms. These guarantees explain why the induced polyhedral decision boundaries remain sharp and scale-invariant, rather than smoothed by Softmax. Empirically, we show that Tropical Attention delivers stronger out-of-distribution generalization in both length and value, with high robustness against perturbative noise, and substantially faster inference with fewer parameters compared to Softmax-based and recurrent attention baselines. For the first time, we extend neural algorithmic reasoning beyond PTIME problems to NP-hard and NP-complete problems, paving the way toward sharper and more expressive Large Reasoning Models (LRMs) capable of tackling complex combinatorial challenges in phylogenetics, cryptography, particle physics, and mathematical discovery.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Identification and Debiased Learning of Causal Effects with General Instrumental Variables
**Date:** 2025-10-24 | **Arxiv:** [2510.20404](https://arxiv.org/abs/2510.20404)

#### Abstract
Instrumental variable methods are fundamental to causal inference when treatment assignment is confounded by unobserved variables. In this article, we develop a general nonparametric framework for identification and learning with multi-categorical or continuous instrumental variables. Specifically, we propose an additive instrumental variable framework to identify mean potential outcomes and the average treatment effect with a weighting function. Leveraging semiparametric theory, we derive efficient influence functions and construct consistent, asymptotically normal estimators via debiased machine learning. Extensions to longitudinal data, dynamic treatment regimes, and multiplicative instrumental variables are further developed. We demonstrate the proposed method by employing simulation studies and analyzing real data from the Job Training Partnership Act program.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Neural Reasoning for Robust Instance Retrieval in $\mathcal{SHOIQ}$
**Date:** 2025-10-24 | **Arxiv:** [2510.20457](https://arxiv.org/abs/2510.20457)

#### Abstract
Concept learning exploits background knowledge in the form of description logic axioms to learn explainable classification models from knowledge bases. Despite recent breakthroughs in neuro-symbolic concept learning, most approaches still cannot be deployed on real-world knowledge bases. This is due to their use of description logic reasoners, which are not robust against inconsistencies nor erroneous data. We address this challenge by presenting a novel neural reasoner dubbed EBR. Our reasoner relies on embeddings to approximate the results of a symbolic reasoner. We show that EBR solely requires retrieving instances for atomic concepts and existential restrictions to retrieve or approximate the set of instances of any concept in the description logic $\mathcal{SHOIQ}$. In our experiments, we compare EBR with state-of-the-art reasoners. Our results suggest that EBR is robust against missing and erroneous data in contrast to existing reasoners.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### InvarGC: Invariant Granger Causality for Heterogeneous Interventional Time Series under Latent Confounding
**Date:** 2025-10-23 | **Arxiv:** [2510.19138](https://arxiv.org/abs/2510.19138)

#### Abstract
Granger causality is widely used for causal structure discovery in complex systems from multivariate time series data. Traditional Granger causality tests based on linear models often fail to detect even mild non-linear causal relationships. Therefore, numerous recent studies have investigated non-linear Granger causality methods, achieving improved performance. However, these methods often rely on two key assumptions: causal sufficiency and known interventional targets. Causal sufficiency assumes the absence of latent confounders, yet their presence can introduce spurious correlations. Moreover, real-world time series data usually come from heterogeneous environments, without prior knowledge of interventions. Therefore, in practice, it is difficult to distinguish intervened environments from non-intervened ones, and even harder to identify which variables or timesteps are affected. To address these challenges, we propose Invariant Granger Causality (InvarGC), which leverages cross-environment heterogeneity to mitigate the effects of latent confounding and to distinguish intervened from non-intervened environments with edge-level granularity, thereby recovering invariant causal relations. In addition, we establish the identifiability under these conditions. Extensive experiments on both synthetic and real-world datasets demonstrate the competitive performance of our approach compared to state-of-the-art methods.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, these methods often rely on two key assumptions: causal sufficiency and known interventional targets.
* **Signal Tags:** #ai

---


### Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization: Bridging Observational and Experimental Data
**Date:** 2025-10-23 | **Arxiv:** [2510.19517](https://arxiv.org/abs/2510.19517)

#### Abstract
Online Internet platforms require sophisticated marketing strategies to optimize user retention and platform revenue -- a classical resource allocation problem. Traditional solutions adopt a two-stage pipeline: machine learning (ML) for predicting individual treatment effects to marketing actions, followed by operations research (OR) optimization for decision-making. This paradigm presents two fundamental technical challenges. First, the prediction-decision misalignment: Conventional ML methods focus solely on prediction accuracy without considering downstream optimization objectives, leading to improved predictive metrics that fail to translate to better decisions. Second, the bias-variance dilemma: Observational data suffers from multiple biases (e.g., selection bias, position bias), while experimental data (e.g., randomized controlled trials), though unbiased, is typically scarce and costly -- resulting in high-variance estimates. We propose Bi-level Decision-Focused Causal Learning (Bi-DFCL) that systematically addresses these challenges. First, we develop an unbiased estimator of OR decision quality using experimental data, which guides ML model training through surrogate loss functions that bridge discrete optimization gradients. Second, we establish a bi-level optimization framework that jointly leverages observational and experimental data, solved via implicit differentiation. This novel formulation enables our unbiased OR estimator to correct learning directions from biased observational data, achieving optimal bias-variance tradeoff. Extensive evaluations on public benchmarks, industrial marketing datasets, and large-scale online A/B tests demonstrate the effectiveness of Bi-DFCL, showing statistically significant improvements over state-of-the-art. Currently, Bi-DFCL has been deployed at Meituan, one of the largest online food delivery platforms in the world.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Who cuts emissions, who turns up the heat? causal machine learning estimates of energy efficiency interventions
**Date:** 2025-10-22 | **Arxiv:** [2508.04478](https://arxiv.org/abs/2508.04478)

#### Abstract
Reducing domestic energy demand is central to climate mitigation and fuel poverty strategies, yet the impact of energy efficiency interventions is highly heterogeneous. Using a causal machine learning model trained on nationally representative data of the English housing stock, we estimate average and conditional treatment effects of wall insulation on gas consumption, focusing on distributional effects across energy burden subgroups. While interventions reduce gas demand on average (by as much as 19 percent), low energy burden groups achieve substantial savings, whereas those experiencing high energy burdens see little to no reduction. This pattern reflects a behaviourally-driven mechanism: households constrained by high costs-to-income ratios (e.g. more than 0.1) reallocate savings toward improved thermal comfort rather than lowering consumption. Far from wasteful, such responses represent rational adjustments in contexts of prior deprivation, with potential co-benefits for health and well-being. These findings call for a broader evaluation framework that accounts for both climate impacts and the equity implications of domestic energy policy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Measure-Theoretic Anti-Causal Representation Learning
**Date:** 2025-10-22 | **Arxiv:** [2510.18052](https://arxiv.org/abs/2510.18052)

#### Abstract
Causal representation learning in the anti-causal setting (labels cause features rather than the reverse) presents unique challenges requiring specialized approaches. We propose Anti-Causal Invariant Abstractions (ACIA), a novel measure-theoretic framework for anti-causal representation learning. ACIA employs a two-level design, low-level representations capture how labels generate observations, while high-level representations learn stable causal patterns across environment-specific variations. ACIA addresses key limitations of existing approaches by accommodating prefect and imperfect interventions through interventional kernels, eliminating dependency on explicit causal structures, handling high-dimensional data effectively, and providing theoretical guarantees for out-of-distribution generalization. Experiments on synthetic and real-world medical datasets demonstrate that ACIA consistently outperforms state-of-the-art methods in both accuracy and invariance metrics. Furthermore, our theoretical results establish tight bounds on performance gaps between training and unseen environments, confirming the efficacy of our approach for robust anti-causal learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Cog-Rethinker: Hierarchical Metacognitive Reinforcement Learning for LLM Reasoning
**Date:** 2025-10-21 | **Arxiv:** [2510.15979](https://arxiv.org/abs/2510.15979)

#### Abstract
Contemporary progress in large language models (LLMs) has revealed notable inferential capacities via reinforcement learning (RL) employing verifiable reward, facilitating the development of O1 and R1-like reasoning models. Directly training from base models with RL is called zero-RL. However, previous works rely upon activating LLMs' inherent capacities through fixed prompt templates. This strategy introduces substantial sampling inefficiencies for weak LLMs, as the majority of problems generate invalid outputs during accuracy-driven filtration in reasoning tasks, which causes a waste of samples. To solve this issue, we propose Cog-Rethinker, a novel hierarchical metacognitive RL framework for LLM reasoning. Our Cog-Rethinker mainly focuses on the rollout procedure in RL training. After the direct rollout, our Cog-Rethinker improves sample utilization in a hierarchical metacognitive two-stage framework. By leveraging human cognition during solving problems, firstly, it prompts policy to decompose zero-accuracy problems into subproblems to produce final reasoning results. Secondly, with zero-accuracy problems in previous rollout stage, it further prompts policy to refine these answers by referencing previous wrong solutions. Moreover, to enable cold-start of the two new reasoning patterns and maintain train-test consistency across prompt templates, our Cog-Rethinker applies supervised fine-tuning on the policy using correct samples of the two stages with direct rollout template. Experimental results demonstrate Cog-Rethinker's superior performance on various mathematical reasoning benchmarks, we also analyzed its improved sample efficiency that accelerates convergence compared to baseline methods.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, previous works rely upon activating LLMs' inherent capacities through fixed prompt templates.
* **Signal Tags:** #ai

---


### Humanoid-inspired Causal Representation Learning for Domain Generalization
**Date:** 2025-10-21 | **Arxiv:** [2510.16382](https://arxiv.org/abs/2510.16382)

#### Abstract
This paper proposes the Humanoid-inspired Structural Causal Model (HSCM), a novel causal framework inspired by human intelligence, designed to overcome the limitations of conventional domain generalization models. Unlike approaches that rely on statistics to capture data-label dependencies and learn distortion-invariant representations, HSCM replicates the hierarchical processing and multi-level learning of human vision systems, focusing on modeling fine-grained causal mechanisms. By disentangling and reweighting key image attributes such as color, texture, and shape, HSCM enhances generalization across diverse domains, ensuring robust performance and interpretability. Leveraging the flexibility and adaptability of human intelligence, our approach enables more effective transfer and learning in dynamic, complex environments. Through both theoretical and empirical evaluations, we demonstrate that HSCM outperforms existing domain generalization models, providing a more principled method for capturing causal relationships and improving model robustness. The code is available at https://github.com/lambett/HSCM.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### REX: Causal discovery based on machine learning and explainability techniques
**Date:** 2025-10-17 | **Arxiv:** [2501.12706](https://arxiv.org/abs/2501.12706)

#### Abstract
Explainable Artificial Intelligence (XAI) techniques hold significant potential for enhancing the causal discovery process, which is crucial for understanding complex systems in areas like healthcare, economics, and artificial intelligence. However, no causal discovery methods currently incorporate explainability into their models to derive the causal graphs. Thus, in this paper we explore this innovative approach, as it offers substantial potential and represents a promising new direction worth investigating. Specifically, we introduce ReX, a causal discovery method that leverages machine learning (ML) models coupled with explainability techniques, specifically Shapley values, to identify and interpret significant causal relationships among variables. Comparative evaluations on synthetic datasets comprising continuous tabular data reveal that ReX outperforms state-of-the-art causal discovery methods across diverse data generation processes, including non-linear and additive noise models. Moreover, ReX was tested on the Sachs single-cell protein-signaling dataset, achieving a precision of 0.952 and recovering key causal relationships with no incorrect edges. Taking together, these results showcase ReX's effectiveness in accurately recovering true causal structures while minimizing false positive predictions, its robustness across diverse datasets, and its applicability to real-world problems. By combining ML and explainability techniques with causal discovery, ReX bridges the gap between predictive modeling and causal inference, offering an effective tool for understanding complex causal structures.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, no causal discovery methods currently incorporate explainability into their models to derive the causal graphs.
* **Signal Tags:** #ai

---


### Causal Disentanglement Learning for Accurate Anomaly Detection in Multivariate Time Series
**Date:** 2025-10-15 | **Arxiv:** [2510.11084](https://arxiv.org/abs/2510.11084)

#### Abstract
Disentangling complex causal relationships is important for accurate detection of anomalies. In multivariate time series analysis, dynamic interactions among data variables over time complicate the interpretation of causal relationships. Traditional approaches assume statistical independence between variables in unsupervised settings, whereas recent methods capture feature correlations through graph representation learning. However, their representations fail to explicitly infer the causal relationships over different time periods. To solve the problem, we propose Causally Disentangled Representation Learning for Anomaly Detection (CDRL4AD) to detect anomalies and identify their causal relationships in multivariate time series. First, we design the causal process as model input, the temporal heterogeneous graph, and causal relationships. Second, our representation identifies causal relationships over different time periods and disentangles latent variables to infer the corresponding causal factors. Third, our experiments on real-world datasets demonstrate that CDRL4AD outperforms state-of-the-art methods in terms of accuracy and root cause analysis. Fourth, our model analysis validates hyperparameter sensitivity and the time complexity of CDRL4AD. Last, we conduct a case study to show how our approach assists human experts in diagnosing the root causes of anomalies.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, their representations fail to explicitly infer the causal relationships over different time periods.
* **Signal Tags:** #ai

---


### LiveThinking: Enabling Real-Time Efficient Reasoning for AI-Powered Livestreaming via Reinforcement Learning
**Date:** 2025-10-10 | **Arxiv:** [2510.07685](https://arxiv.org/abs/2510.07685)

#### Abstract
In AI-powered e-commerce livestreaming, digital avatars require real-time responses to drive engagement, a task for which high-latency Large Reasoning Models (LRMs) are ill-suited. We introduce LiveThinking, a practical two-stage optimization framework to bridge this gap. First, we address computational cost by distilling a 670B teacher LRM into a lightweight 30B Mixture-of-Experts (MoE) model (3B active) using Rejection Sampling Fine-Tuning (RFT). This reduces deployment overhead but preserves the teacher's verbose reasoning, causing latency. To solve this, our second stage employs reinforcement learning with Group Relative Policy Optimization (GRPO) to compress the model's reasoning path, guided by a multi-objective reward function balancing correctness, helpfulness, and brevity. LiveThinking achieves a 30-fold reduction in computational cost, enabling sub-second latency. In real-world application on Taobao Live, it improved response correctness by 3.3% and helpfulness by 21.8%. Tested by hundreds of thousands of viewers, our system led to a statistically significant increase in Gross Merchandise Volume (GMV), demonstrating its effectiveness in enhancing user experience and commercial performance in live, interactive settings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Unlocking Reasoning Capabilities in LLMs via Reinforcement Learning Exploration
**Date:** 2025-10-07 | **Arxiv:** [2510.03865](https://arxiv.org/abs/2510.03865)

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has recently enhanced the reasoning capabilities of large language models (LLMs), particularly for mathematical problem solving. However, a fundamental limitation remains: as the sampling budget increases, the advantage of RLVR-trained models over their pretrained bases often diminishes or even vanishes, revealing a strong dependence on the base model's restricted search space. We attribute this phenomenon to the widespread use of the reverse Kullback-Leibler (KL) divergence regularizer, whose mode-seeking behavior keeps the policy trapped inside the base model's support region and hampers wider exploration. To address this issue, we propose RAPO (Rewards-Aware Policy Optimization), an algorithm to promote broader yet focused exploration. Our method (i) utilizes the forward KL penalty to replace the reverse KL penalty for out-of-distribution exploration, and (ii) reweights the reference policy to facilitate adaptive in-distribution exploration. We train Qwen2.5-3B and 7B models with RAPO on the 8K SimpleRL-Zero dataset, without supervised fine-tuning, and evaluate them on AIME2024 and AIME2025. Results show that RAPO consistently improves problem-solving performance. Notably, RAPO enables models to surpass the base model's performance ceiling and solves previously intractable problems, advancing the frontier of RLVR for challenging reasoning tasks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, a fundamental limitation remains: as the sampling budget increases, the advantage of RLVR-trained models over their pretrained bases often diminishes or even vanishes, revealing a strong dependence on the base model's restricted search space.
* **Signal Tags:** #ai

---


### RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning
**Date:** 2025-10-06 | **Arxiv:** [2510.02892](https://arxiv.org/abs/2510.02892)

#### Abstract
Reinforcement learning (RL) is central to improving reasoning in large language models (LLMs) but typically requires ground-truth rewards. Test-Time Reinforcement Learning (TTRL) removes this need by using majority-vote rewards, but relies on heavy online RL and incurs substantial computational cost. We propose RoiRL: Reasoning with offline iterative Reinforcement Learning, a family of lightweight offline learning alternatives that can target the same regularized optimal policies. Unlike TTRL, RoiRL eliminates the need to maintain a reference model and instead optimizes weighted log-likelihood objectives, enabling stable training with significantly lower memory and compute requirements. Experimental results show that RoiRL trains to 2.5x faster and consistently outperforms TTRL on reasoning benchmarks, establishing a scalable path to self-improving LLMs without labels.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Geo-R1: Unlocking VLM Geospatial Reasoning with Cross-View Reinforcement Learning
**Date:** 2025-10-02 | **Arxiv:** [2510.00072](https://arxiv.org/abs/2510.00072)

#### Abstract
We introduce Geo-R1, a reasoning-centric post-training framework that unlocks geospatial reasoning in vision-language models by combining thinking scaffolding and elevating. In the scaffolding stage, Geo-R1 instills a ``geospatial thinking paradigm" via supervised fine-tuning on synthetic chain-of-thought exemplars, enabling models to connect visual cues with geographic priors without costly human reasoning annotations. In the elevating stage, it uses GRPO-based reinforcement learning on a weakly-supervised cross-view pairing proxy. This design supplies a verifiable and scalable reward signal: teaching models to capture and reconcile features across modalities, and harnessing reasoning for accurate prediction. Geo-R1 extends geospatial modeling from domain pretraining / supervised finetuning to reasoning-first post-training, and achieves state-of-the-art performance across various geospatial reasoning benchmarks. Our model is available at https://huggingface.co/miniHui/Geo-R1.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Eliciting Chain-of-Thought Reasoning for Time Series Analysis using Reinforcement Learning
**Date:** 2025-10-02 | **Arxiv:** [2510.01116](https://arxiv.org/abs/2510.01116)

#### Abstract
Complex numerical time series analysis often demands multi-step reasoning capabilities beyond current models' reach. Tasks like medical diagnosis and weather forecasting require sequential reasoning processes -- including counterfactual analysis, logical deduction, knowledge application, and multi-modal contextual integration -- that existing time series models cannot explicitly perform. While recent research has shown large language models (LLMs) can achieve sophisticated Chain-of-Thought (CoT) reasoning through reinforcement learning (RL), these advances have primarily focused on mathematical and coding domains, with LLMs still demonstrating poor performance on time series tasks. We introduce Chain Of thought for Understanding Numerical Time Series (COUNTS), the first framework that trains LLMs to perform CoT reasoning across diverse time series tasks using RL with verifiable rewards. Our approach employs a Residual Vector-Quantized VAE to create high-fidelity discrete tokens that seamlessly integrate into a pre-trained LLM's vocabulary. COUNTS undergoes a two-stage training process: first, supervised fine-tuning on time series analysis tasks to master our novel representations, followed by Group Relative Policy Optimization training on verifiable problems using prompting strategies that encourage explicit reasoning steps before producing final answers. Our experiments demonstrate that this RL-driven approach with intermediate CoT reasoning significantly enhances LLM performance across various time series analysis tasks, opening new possibilities for complex temporal data reasoning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Causal-EPIG: A Prediction-Oriented Active Learning Framework for CATE Estimation
**Date:** 2025-09-29 | **Arxiv:** [2509.21866](https://arxiv.org/abs/2509.21866)

#### Abstract
Estimating the Conditional Average Treatment Effect (CATE) is often constrained by the high cost of obtaining outcome measurements, making active learning essential. However, conventional active learning strategies suffer from a fundamental objective mismatch. They are designed to reduce uncertainty in model parameters or in observable factual outcomes, failing to directly target the unobservable causal quantities that are the true objects of interest. To address this misalignment, we introduce the principle of causal objective alignment, which posits that acquisition functions should target unobservable causal quantities, such as the potential outcomes and the CATE, rather than indirect proxies. We operationalize this principle through the Causal-EPIG framework, which adapts the information-theoretic criterion of Expected Predictive Information Gain (EPIG) to explicitly quantify the value of a query in terms of reducing uncertainty about unobservable causal quantities. From this unified framework, we derive two distinct strategies that embody a fundamental trade-off: a comprehensive approach that robustly models the full causal mechanisms via the joint potential outcomes, and a focused approach that directly targets the CATE estimand for maximum sample efficiency. Extensive experiments demonstrate that our strategies consistently outperform standard baselines, and crucially, reveal that the optimal strategy is context-dependent, contingent on the base estimator and data complexity. Our framework thus provides a principled guide for sample-efficient CATE estimation in practice.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, conventional active learning strategies suffer from a fundamental objective mismatch.
* **Signal Tags:** #ai

---


### Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs
**Date:** 2025-09-03 | **Arxiv:** [2509.00084](https://arxiv.org/abs/2509.00084)

#### Abstract
To further enhance the ability of Large Language Models (LLMs) to solve complex, multi-step reasoning problems, test-time scaling (TTS) methods have gained widespread attention. Existing approaches such as Best-of-N and majority voting are limited as their performance depends on the quality of candidate responses, making them unable to produce a correct solution when all candidates are incorrect. Introducing an additional model to select the best response also incurs significant deployment costs. To this end, we introduce Generative Self-Refinement (GSR), a novel parallel test-time scaling framework where a unified model first generates a set of candidate responses in parallel and then performs self-refinement to synthesize a new superior solution based on a prompt consisting of the problem and these candidates. However, LLMs struggle to perform refinement effectively when prompted directly. Therefore, we design a hybrid training pipeline by jointly optimizing for two complementary objectives, solving problems directly and refining candidate responses. Experimental results demonstrate that our method achieves state-of-the-art performance across five mathematical benchmarks. We further show that this learned self-refinement skill is a model-agnostic enhancement, robust across different model scales and generalizing to out-of-distribution reasoning tasks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, LLMs struggle to perform refinement effectively when prompted directly.
* **Signal Tags:** #ai

---


### Learning Marked Temporal Point Process Explanations based on Counterfactual and Factual Reasoning
**Date:** 2025-08-19 | **Arxiv:** [2508.11943](https://arxiv.org/abs/2508.11943)

#### Abstract
Neural network-based Marked Temporal Point Process (MTPP) models have been widely adopted to model event sequences in high-stakes applications, raising concerns about the trustworthiness of outputs from these models. This study focuses on Explanation for MTPP, aiming to identify the minimal and rational explanation, that is, the minimum subset of events in history, based on which the prediction accuracy of MTPP matches that based on full history to a great extent and better than that based on the complement of the subset. This study finds that directly defining Explanation for MTPP as counterfactual explanation or factual explanation can result in irrational explanations. To address this issue, we define Explanation for MTPP as a combination of counterfactual explanation and factual explanation. This study proposes Counterfactual and Factual Explainer for MTPP (CFF) to solve Explanation for MTPP with a series of deliberately designed techniques. Experiments demonstrate the correctness and superiority of CFF over baselines regarding explanation quality and processing efficiency.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### A Multi-Resolution Benchmark Framework for Spatial Reasoning Assessment in Neural Networks
**Date:** 2025-08-19 | **Arxiv:** [2508.12741](https://arxiv.org/abs/2508.12741)

#### Abstract
This paper presents preliminary results in the definition of a comprehensive benchmark framework designed to systematically evaluate spatial reasoning capabilities in neural networks, with a particular focus on morphological properties such as connectivity and distance relationships. The framework is currently being used to study the capabilities of nnU-Net, exploiting the spatial model checker VoxLogicA to generate two distinct categories of synthetic datasets: maze connectivity problems for topological analysis and spatial distance computation tasks for geometric understanding. Each category is evaluated across multiple resolutions to assess scalability and generalization properties. The automated pipeline encompasses a complete machine learning workflow including: synthetic dataset generation, standardized training with cross-validation, inference execution, and comprehensive evaluation using Dice coefficient and IoU (Intersection over Union) metrics. Preliminary experimental results demonstrate significant challenges in neural network spatial reasoning capabilities, revealing systematic failures in basic geometric and topological understanding tasks. The framework provides a reproducible experimental protocol, enabling researchers to identify specific limitations. Such limitations could be addressed through hybrid approaches combining neural networks with symbolic reasoning methods for improved spatial understanding in clinical applications, establishing a foundation for ongoing research into neural network spatial reasoning limitations and potential solutions.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Train Long, Think Short: Curriculum Learning for Efficient Reasoning
**Date:** 2025-08-13 | **Arxiv:** [2508.08940](https://arxiv.org/abs/2508.08940)

#### Abstract
Recent work on enhancing the reasoning abilities of large language models (LLMs) has introduced explicit length control as a means of constraining computational cost while preserving accuracy. However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning. In this work, we propose a curriculum learning strategy for length-controlled reasoning using Group Relative Policy Optimization (GRPO). Our method starts with generous token budgets and gradually tightens them over training, encouraging models to first discover effective solution strategies and then distill them into more concise reasoning traces. We augment GRPO with a reward function that balances three signals: task correctness (via verifier feedback), length efficiency, and formatting adherence (via structural tags). Experiments on GSM8K, MATH500, SVAMP, College Math, and GSM+ demonstrate that curriculum-based training consistently outperforms fixed-budget baselines at the same final budget, achieving higher accuracy and significantly improved token efficiency. We further ablate the impact of reward weighting and decay schedule design, showing that progressive constraint serves as a powerful inductive bias for training efficient reasoning models. Our code and checkpoints are released at: https://github.com/hammoudhasan/curriculum_grpo.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning.
* **Signal Tags:** #ai

---


### Text Rationalization for Robust Causal Effect Estimation
**Date:** 2025-12-08 | **Arxiv:** [2512.05373](https://arxiv.org/abs/2512.05373)

#### Abstract
Recent advances in natural language processing have enabled the increasing use of text data in causal inference, particularly for adjusting confounding factors in treatment effect estimation. Although high-dimensional text can encode rich contextual information, it also poses unique challenges for causal identification and estimation. In particular, the positivity assumption, which requires sufficient treatment overlap across confounder values, is often violated at the observational level, when massive text is represented in feature spaces. Redundant or spurious textual features inflate dimensionality, producing extreme propensity scores, unstable weights, and inflated variance in effect estimates. We address these challenges with Confounding-Aware Token Rationalization (CATR), a framework that selects a sparse necessary subset of tokens using a residual-independence diagnostic designed to preserve confounding information sufficient for unconfoundedness. By discarding irrelevant texts while retaining key signals, CATR mitigates observational-level positivity violations and stabilizes downstream causal effect estimators. Experiments on synthetic data and a real-world study using the MIMIC-III database demonstrate that CATR yields more accurate, stable, and interpretable causal effect estimates than existing baselines.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Arbitrage: Efficient Reasoning via Advantage-Aware Speculation
**Date:** 2025-12-05 | **Arxiv:** [2512.05033](https://arxiv.org/abs/2512.05033)

#### Abstract
Modern Large Language Models achieve impressive reasoning capabilities with long Chain of Thoughts, but they incur substantial computational cost during inference, and this motivates techniques to improve the performance-cost ratio. Among these techniques, Speculative Decoding accelerates inference by employing a fast but inaccurate draft model to autoregressively propose tokens, which are then verified in parallel by a more capable target model. However, due to unnecessary rejections caused by token mismatches in semantically equivalent steps, traditional token-level Speculative Decoding struggles in reasoning tasks. Although recent works have shifted to step-level semantic verification, which improve efficiency by accepting or rejecting entire reasoning steps, existing step-level methods still regenerate many rejected steps with little improvement, wasting valuable target compute. To address this challenge, we propose Arbitrage, a novel step-level speculative generation framework that routes generation dynamically based on the relative advantage between draft and target models. Instead of applying a fixed acceptance threshold, Arbitrage uses a lightweight router trained to predict when the target model is likely to produce a meaningfully better step. This routing approximates an ideal Arbitrage Oracle that always chooses the higher-quality step, achieving near-optimal efficiency-accuracy trade-offs. Across multiple mathematical reasoning benchmarks, Arbitrage consistently surpasses prior step-level Speculative Decoding baselines, reducing inference latency by up to $\sim2\times$ at matched accuracy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, due to unnecessary rejections caused by token mismatches in semantically equivalent steps, traditional token-level Speculative Decoding struggles in reasoning tasks.
* **Signal Tags:** #ai

---


### A Fast Kernel-based Conditional Independence test with Application to Causal Discovery
**Date:** 2025-12-05 | **Arxiv:** [2505.11085](https://arxiv.org/abs/2505.11085)

#### Abstract
Kernel-based conditional independence (KCI) testing is a powerful nonparametric method commonly employed in causal discovery tasks. Despite its flexibility and statistical reliability, cubic computational complexity limits its application to large datasets. To address this computational bottleneck, we propose \textit{FastKCI}, a scalable and parallelizable kernel-based conditional independence test that utilizes a mixture-of-experts approach inspired by embarrassingly parallel inference techniques for Gaussian processes. By partitioning the dataset based on a Gaussian mixture model over the conditioning variables, FastKCI conducts local KCI tests in parallel, aggregating the results using an importance-weighted sampling scheme. Experiments on synthetic datasets and benchmarks on real-world production data validate that FastKCI maintains the statistical power of the original KCI test while achieving substantial computational speedups. FastKCI thus represents a practical and efficient solution for conditional independence testing in causal inference on large-scale data.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### REM: Evaluating LLM Embodied Spatial Reasoning through Multi-Frame Trajectories
**Date:** 2025-12-02 | **Arxiv:** [2512.00736](https://arxiv.org/abs/2512.00736)

#### Abstract
Humans build viewpoint-independent cognitive maps through navigation, enabling intuitive reasoning about object permanence and spatial relations. We argue that multimodal large language models (MLLMs), despite extensive video training, lack this fundamental spatial reasoning capability, a critical limitation for embodied applications. To demonstrate these limitations and drive research, we introduce REM (Reasoning over Embodied Multi-Frame Trajectories), a benchmark using controllable 3D environments for long-horizon embodied spatial reasoning. REM systematically evaluates key aspects like object permanence/distinction, spatial relationships, and numerical tracking across dynamic embodied viewpoints. Our evaluation shows that the best-performing current models exhibit promising overall performance, but become increasingly unreliable at even moderate complexity levels easily handled by humans. These findings highlight challenges MLLMs face in developing robust spatial representations from sequential visual input. Consequently, REM provides targeted metrics and diagnostics to foster improved spatial understanding in future models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning
**Date:** 2025-11-26 | **Arxiv:** [2502.07154](https://arxiv.org/abs/2502.07154)

#### Abstract
Recent progress in large language models (LLMs) highlights the power of scaling test-time compute to achieve strong performance on complex tasks, such as mathematical reasoning and code generation. This raises a critical question: how should model training be modified to optimize performance under a subsequent test-time compute strategy and budget? To explore this, we focus on pass@N, a simple test-time strategy that searches for a correct answer in $N$ independent samples. We show, surprisingly, that training with cross-entropy (CE) loss can be ${\it misaligned}$ with pass@N in that pass@N accuracy ${\it decreases}$ with longer training. We explain the origins of this misalignment in terms of model overconfidence induced by CE, and experimentally verify our prediction of overconfidence as an impediment to scaling test-time compute via pass@N. Furthermore we suggest a principled, modified training loss that is better aligned to pass@N by limiting model confidence and rescuing pass@N test performance. Our algorithm demonstrates improved mathematical reasoning on MATH and MiniF2F benchmarks under several scenarios: (1) providing answers to math questions; and (2) proving theorems by searching over proof trees of varying shapes. Overall our work underscores the importance of co-designing two traditionally separate phases of LLM development: training-time protocols and test-time search and reasoning strategies.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning
**Date:** 2025-11-26 | **Arxiv:** [2511.19942](https://arxiv.org/abs/2511.19942)

#### Abstract
It is widely recognized that reinforcement learning (RL) fine-tuning of large language models often leads to diversity collapse, where outputs lack variety. Prior work has proposed a range of heuristics to counteract this effect, but these methods are ad hoc: they frequently trade off correctness for diversity, their effectiveness varies across tasks, and in some cases they even contradict one another. In this work, we place these observations on a rigorous foundation. We first provide a formal proof of why RL fine-tuning exhibits diversity collapse via a selection and reinforcement bias. Next, we make a key observation that any reward modification to address diversity collapse only needs to be applied on the correct trajectories. Building directly on this analysis, we introduce a principled method -- differential smoothing -- that provably improves both correctness and diversity, outperforming vanilla RL as well as widely used entropy-based heuristics. Our theory precisely characterizes when existing heuristics help and why they fail, while showing that differential smoothing is universally superior. Extensive experiments with models from 1B to 7B parameters, across domains including CountDown and real-world mathematical reasoning, demonstrate consistent gains. Differential smoothing improves both Pass@1 and Pass@k, with up to 6.7% improvements on AIME24 dataset.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Unified Non-Convex Framework for Robust Causal Inference: Overcoming the Gaussian Barrier and Optimization Fragility
**Date:** 2025-11-25 | **Arxiv:** [2511.19284](https://arxiv.org/abs/2511.19284)

#### Abstract
This document proposes a Unified Robust Framework that re-engineers the estimation of the Average Treatment Effect on the Overlap (ATO). It synthesizes gamma-Divergence for outlier robustness, Graduated Non-Convexity (GNC) for global optimization, and a "Gatekeeper" mechanism to address the impossibility of higher-order orthogonality in Gaussian regimes.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
