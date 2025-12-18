# Vol 11 Causal Inference   Logic
*Enriched by BITCOREOS | Phase 4 Batch 3*

---

### On Transportability for Structural Causal Bandits
**Date:** 2025-11-25 | **Arxiv:** [2511.17953](https://hub.bitwiki.org/t/on-transportability-for-structural-causal-bandits/25376)

#### Abstract
Intelligent agents equipped with causal knowledge can optimize their action spaces to avoid unnecessary exploration. The structural causal bandit framework provides a graphical characterization for identifying actions that are unable to maximize rewards by leveraging prior knowledge of the underlying causal structure. While such knowledge enables an agent to estimate the expected rewards of certain actions based on others in online interactions, there has been little guidance on how to transfer information inferred from arbitrary combinations of datasets collected under different conditions -- observational or experimental -- and from heterogeneous environments. In this paper, we investigate the structural causal bandit with transportability, where priors from the source environments are fused to enhance learning in the deployment setting. We demonstrate that it is possible to exploit invariances across environments to consistently improve learning. The resulting bandit algorithm achieves a sub-linear regret bound with an explicit dependence on informativeness of prior data, and it may outperform standard bandit approaches that rely solely on online learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Causal Intervention Sequence Analysis for Fault Tracking in Radio Access Networks
**Date:** 2025-11-25 | **Arxiv:** [2511.17505](https://hub.bitwiki.org/t/causal-intervention-sequence-analysis-for-fault-tracking-in-radio-access-networks/25570)

#### Abstract
To keep modern Radio Access Networks (RAN) running smoothly, operators need to spot the real-world triggers behind Service-Level Agreement (SLA) breaches well before customers feel them. We introduce an AI/ML pipeline that does two things most tools miss: (1) finds the likely root-cause indicators and (2) reveals the exact order in which those events unfold. We start by labeling network data: records linked to past SLA breaches are marked `abnormal', and everything else `normal'. Our model then learns the causal chain that turns normal behavior into a fault. In Monte Carlo tests the approach pinpoints the correct trigger sequence with high precision and scales to millions of data points without loss of speed. These results show that high-resolution, causally ordered insights can move fault management from reactive troubleshooting to proactive prevention.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter
**Date:** 2025-11-21 | **Arxiv:** [2511.16665](https://hub.bitwiki.org/t/taming-the-long-tail-efficient-reasoning-rl-training-with-adaptive-drafter/25013)

#### Abstract
The emergence of Large Language Models (LLMs) with strong reasoning capabilities marks a significant milestone, unlocking new frontiers in complex problem-solving. However, training these reasoning models, typically using Reinforcement Learning (RL), encounters critical efficiency bottlenecks: response generation during RL training exhibits a persistent long-tail distribution, where a few very long responses dominate execution time, wasting resources and inflating costs. To address this, we propose TLT, a system that accelerates reasoning RL training losslessly by integrating adaptive speculative decoding. Applying speculative decoding in RL is challenging due to the dynamic workloads, evolving target model, and draft model training overhead. TLT overcomes these obstacles with two synergistic components: (1) Adaptive Drafter, a lightweight draft model trained continuously on idle GPUs during long-tail generation to maintain alignment with the target model at no extra cost; and (2) Adaptive Rollout Engine, which maintains a memory-efficient pool of pre-captured CUDAGraphs and adaptively select suitable SD strategies for each input batch. Evaluations demonstrate that TLT achieves over 1.7x end-to-end RL training speedup over state-of-the-art systems, preserves the model accuracy, and yields a high-quality draft model as a free byproduct suitable for efficient deployment. Code is released at https://github.com/mit-han-lab/fastrl.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, training these reasoning models, typically using Reinforcement Learning (RL), encounters critical efficiency bottlenecks: response generation during RL training exhibits a persistent long-tail distribution, where a few very long responses dominate execution time, wasting resources and inflating costs.
* **Signal Tags:** #ai

---


### Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization
**Date:** 2025-11-20 | **Arxiv:** [2511.14846](https://hub.bitwiki.org/t/empowering-multi-turn-tool-integrated-reasoning-with-group-turn-policy-optimization/24732)

#### Abstract
Training Large Language Models (LLMs) for multi-turn Tool-Integrated Reasoning (TIR) - where models iteratively reason, generate code, and verify through execution - remains challenging for existing reinforcement learning (RL) approaches. Current RL methods, exemplified by Group Relative Policy Optimization (GRPO), suffer from coarse-grained, trajectory-level rewards that provide insufficient learning signals for complex multi-turn interactions, leading to training stagnation. To address this issue, we propose Group Turn Policy Optimization (GTPO), a novel RL algorithm specifically designed for training LLMs on multi-turn TIR tasks. GTPO introduces three key innovations: (1) turn-level reward assignment that provides fine-grained feedback for individual turns, (2) return-based advantage estimation where normalized discounted returns are calculated as advantages, and (3) self-supervised reward shaping that exploits self-supervision signals from generated code to densify sparse binary outcome-based rewards. Our comprehensive evaluation demonstrates that GTPO outperforms GRPO by 3.0% on average across diverse reasoning benchmarks, establishing its effectiveness for advancing complex mathematical reasoning in the real world.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Explaining Time Series Classification Predictions via Causal Attributions
**Date:** 2025-11-20 | **Arxiv:** [2405.15871](https://hub.bitwiki.org/t/explaining-time-series-classification-predictions-via-causal-attributions/24771)

#### Abstract
Despite the excelling performance of machine learning models, understanding their decisions remains a long-standing goal. Although commonly used attribution methods from explainable AI attempt to address this issue, they typically rely on associational rather than causal relationships. In this study, within the context of time series classification, we introduce a novel model-agnostic attribution method to assess the causal effect of concepts i.e., predefined segments within a time series, on classification outcomes. Our approach compares these causal attributions with closely related associational attributions, both theoretically and empirically. To estimate counterfactual outcomes, we use state-of-the-art diffusion models backed by state space models. We demonstrate the insights gained by our approach for a diverse set of qualitatively different time series classification tasks. Although causal and associational attributions might often share some similarities, in all cases they differ in important details, underscoring the risks associated with drawing causal conclusions from associational data alone. We believe that the proposed approach is also widely applicable in other domains to shed some light on the limits of associational attributions.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### From Solving to Verifying: A Unified Objective for Robust Reasoning in LLMs
**Date:** 2025-11-20 | **Arxiv:** [2511.15137](https://hub.bitwiki.org/t/from-solving-to-verifying-a-unified-objective-for-robust-reasoning-in-llms/24768)

#### Abstract
The reasoning capabilities of large language models (LLMs) have been significantly improved through reinforcement learning (RL). Nevertheless, LLMs still struggle to consistently verify their own reasoning traces. This raises the research question of how to enhance the self-verification ability of LLMs and whether such an ability can further improve reasoning performance. In this work, we propose GRPO-Verif, an algorithm that jointly optimizes solution generation and self-verification within a unified loss function, with an adjustable hyperparameter controlling the weight of the verification signal. Experimental results demonstrate that our method enhances self-verification capability while maintaining comparable performance in reasoning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Better LLM Reasoning via Dual-Play
**Date:** 2025-11-18 | **Arxiv:** [2511.11881](https://hub.bitwiki.org/t/better-llm-reasoning-via-dual-play/24066)

#### Abstract
Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at https://hcy123902.github.io/PasoDoble.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Bias-Restrained Prefix Representation Finetuning for Mathematical Reasoning
**Date:** 2025-11-17 | **Arxiv:** [2511.10707](https://hub.bitwiki.org/t/bias-restrained-prefix-representation-finetuning-for-mathematical-reasoning/23687)

#### Abstract
Parameter-Efficient finetuning (PEFT) enhances model performance on downstream tasks by updating a minimal subset of parameters. Representation finetuning (ReFT) methods further improve efficiency by freezing model weights and optimizing internal representations with fewer parameters than PEFT, outperforming PEFT on several tasks. However, ReFT exhibits a significant performance decline on mathematical reasoning tasks. To address this problem, the paper demonstrates that ReFT's poor performance on mathematical tasks primarily stems from its struggle to generate effective reasoning prefixes during the early inference phase. Moreover, ReFT disturbs the numerical encoding and the error accumulats during the CoT stage. Based on these observations, this paper proposes Bias-REstrained Prefix Representation FineTuning (BREP ReFT), which enhances ReFT's mathematical reasoning capability by truncating training data to optimize the generation of initial reasoning prefixes, intervening on the early inference stage to prevent error accumulation, and constraining the intervention vectors' magnitude to avoid disturbing numerical encoding. Extensive experiments across diverse model architectures demonstrate BREP's superior effectiveness, efficiency, and robust generalization capability, outperforming both standard ReFT and weight-based PEFT methods on the task of mathematical reasoning. The source code is available at https://github.com/LiangThree/BREP.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, ReFT exhibits a significant performance decline on mathematical reasoning tasks.
* **Signal Tags:** #ai

---


### Generalizing to Unseen Disaster Events: A Causal View
**Date:** 2025-11-14 | **Arxiv:** [2511.10120](https://hub.bitwiki.org/t/generalizing-to-unseen-disaster-events-a-causal-view/23569)

#### Abstract
Due to the rapid growth of social media platforms, these tools have become essential for monitoring information during ongoing disaster events. However, extracting valuable insights requires real-time processing of vast amounts of data. A major challenge in existing systems is their exposure to event-related biases, which negatively affects their ability to generalize to emerging events. While recent advancements in debiasing and causal learning offer promising solutions, they remain underexplored in the disaster event domain. In this work, we approach bias mitigation through a causal lens and propose a method to reduce event- and domain-related biases, enhancing generalization to future events. Our approach outperforms multiple baselines by up to +1.9% F1 and significantly improves a PLM-based classifier across three disaster classification tasks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, extracting valuable insights requires real-time processing of vast amounts of data.
* **Signal Tags:** #ai

---


### A Causal Framework to Measure and Mitigate Non-binary Treatment Discrimination
**Date:** 2025-11-13 | **Arxiv:** [2503.22454](https://hub.bitwiki.org/t/a-causal-framework-to-measure-and-mitigate-non-binary-treatment-discrimination/23364)

#### Abstract
Fairness studies of algorithmic decision-making systems often simplify complex decision processes, such as bail or loan approvals, into binary classification tasks. However, these approaches overlook that such decisions are not inherently binary (e.g., approve or not approve bail or loan); they also involve non-binary treatment decisions (e.g., bail conditions or loan terms) that can influence the downstream outcomes (e.g., loan repayment or reoffending). In this paper, we argue that non-binary treatment decisions are integral to the decision process and controlled by decision-makers and, therefore, should be central to fairness analyses in algorithmic decision-making. We propose a causal framework that extends fairness analyses and explicitly distinguishes between decision-subjects' covariates and the treatment decisions. This specification allows decision-makers to use our framework to (i) measure treatment disparity and its downstream effects in historical data and, using counterfactual reasoning, (ii) mitigate the impact of past unfair treatment decisions when automating decision-making. We use our framework to empirically analyze four widely used loan approval datasets to reveal potential disparity in non-binary treatment decisions and their discriminatory impact on outcomes, highlighting the need to incorporate treatment decisions in fairness assessments. Moreover, by intervening in treatment decisions, we show that our framework effectively mitigates treatment discrimination from historical data to ensure fair risk score estimation and (non-binary) decision-making processes that benefit all stakeholders.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, these approaches overlook that such decisions are not inherently binary (e.
* **Signal Tags:** #ai

---


### Reasoning on Time-Series for Financial Technical Analysis
**Date:** 2025-11-13 | **Arxiv:** [2511.08616](https://hub.bitwiki.org/t/reasoning-on-time-series-for-financial-technical-analysis/23300)

#### Abstract
While Large Language Models have been used to produce interpretable stock forecasts, they mainly focus on analyzing textual reports but not historical price data, also known as Technical Analysis. This task is challenging as it switches between domains: the stock price inputs and outputs lie in the time-series domain, while the reasoning step should be in natural language. In this work, we introduce Verbal Technical Analysis (VTA), a novel framework that combine verbal and latent reasoning to produce stock time-series forecasts that are both accurate and interpretable. To reason over time-series, we convert stock price data into textual annotations and optimize the reasoning trace using an inverse Mean Squared Error (MSE) reward objective. To produce time-series outputs from textual reasoning, we condition the outputs of a time-series backbone model on the reasoning-based attributes. Experiments on stock datasets across U.S., Chinese, and European markets show that VTA achieves state-of-the-art forecasting accuracy, while the reasoning traces also perform well on evaluation by industry experts.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Vector Symbolic Algebras for the Abstraction and Reasoning Corpus
**Date:** 2025-11-13 | **Arxiv:** [2511.08747](https://hub.bitwiki.org/t/vector-symbolic-algebras-for-the-abstraction-and-reasoning-corpus/23314)

#### Abstract
The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) is a generative, few-shot fluid intelligence benchmark. Although humans effortlessly solve ARC-AGI, it remains extremely difficult for even the most advanced artificial intelligence systems. Inspired by methods for modelling human intelligence spanning neuroscience to psychology, we propose a cognitively plausible ARC-AGI solver. Our solver integrates System 1 intuitions with System 2 reasoning in an efficient and interpretable process using neurosymbolic methods based on Vector Symbolic Algebras (VSAs). Our solver works by object-centric program synthesis, leveraging VSAs to represent abstract objects, guide solution search, and enable sample-efficient neural learning. Preliminary results indicate success, with our solver scoring 10.8% on ARC-AGI-1-Train and 3.0% on ARC-AGI-1-Eval. Additionally, our solver performs well on simpler benchmarks, scoring 94.5% on Sort-of-ARC and 83.1% on 1D-ARC -- the latter outperforming GPT-4 at a tiny fraction of the computational cost. Importantly, our approach is unique; we believe we are the first to apply VSAs to ARC-AGI and have developed the most cognitively plausible ARC-AGI solver yet. Our code is available at: https://github.com/ijoffe/ARC-VSA-2025.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Test-time Diverse Reasoning by Riemannian Activation Steering
**Date:** 2025-11-12 | **Arxiv:** [2511.08305](https://hub.bitwiki.org/t/test-time-diverse-reasoning-by-riemannian-activation-steering/23018)

#### Abstract
Best-of-$N$ reasoning improves the accuracy of language models in solving complex tasks by sampling multiple candidate solutions and then selecting the best one based on some criteria. A critical bottleneck for this strategy is the output diversity limit, which occurs when the model generates similar outputs despite stochastic sampling, and hence recites the same error. To address this lack of variance in reasoning paths, we propose a novel unsupervised activation steering strategy that simultaneously optimizes the steering vectors for multiple reasoning trajectories at test time. At any synchronization anchor along the batch generation process, we find the steering vectors that maximize the total volume spanned by all possible intervened activation subsets. We demonstrate that these steering vectors can be determined by solving a Riemannian optimization problem over the product of spheres with a log-determinant objective function. We then use a Riemannian block-coordinate descent algorithm with a well-tuned learning rate to obtain a stationary point of the problem, and we apply these steering vectors until the generation process reaches the subsequent synchronization anchor. Empirical evaluations on popular mathematical benchmarks demonstrate that our test-time Riemannian activation steering strategy outperforms vanilla sampling techniques in terms of generative diversity and solution accuracy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### DynaSolidGeo: A Dynamic Benchmark for Genuine Spatial Mathematical Reasoning of VLMs in Solid Geometry
**Date:** 2025-11-12 | **Arxiv:** [2510.22340](https://hub.bitwiki.org/t/dynasolidgeo-a-dynamic-benchmark-for-genuine-spatial-mathematical-reasoning-of-vlms-in-solid-geometry/23152)

#### Abstract
Solid geometry problem solving demands spatial mathematical reasoning that integrates spatial intelligence and symbolic reasoning. However, most existing multimodal mathematical reasoning benchmarks focus primarily on 2D plane geometry, rely on static datasets prone to data contamination and memorization, and evaluate models solely by final answers, overlooking the reasoning process. To address these limitations, we introduce DynaSolidGeo, the first dynamic benchmark for evaluating genuine spatial reasoning in Vision-Language Models (VLMs). Constructed through a semi-automatic annotation pipeline, DynaSolidGeo contains 503 expert-curated seed questions that can, in principle, dynamically generate an unbounded number of diverse multimodal text-visual instances. Beyond answer accuracy, we incorporate process evaluation based on expert-annotated reasoning chains to measure logical validity and causal coherence. Experiments across representative open-source and closed-source VLMs reveal large performance gaps, severe degradation in dynamic settings, and poor performance on tasks requiring high-level spatial intelligence, such as mental rotation and visualization. The code and dataset are available at \href{https://zgca-ai4edu.github.io/DynaSolidGeo/}{DynaSolidGeo}.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, most existing multimodal mathematical reasoning benchmarks focus primarily on 2D plane geometry, rely on static datasets prone to data contamination and memorization, and evaluate models solely by final answers, overlooking the reasoning process.
* **Signal Tags:** #ai

---


### Consistency Is Not Always Correct: Towards Understanding the Role of Exploration in Post-Training Reasoning
**Date:** 2025-11-11 | **Arxiv:** [2511.07368](https://hub.bitwiki.org/t/consistency-is-not-always-correct-towards-understanding-the-role-of-exploration-in-post-training-reasoning/22697)

#### Abstract
Foundation models exhibit broad knowledge but limited task-specific reasoning, motivating post-training strategies such as RLVR and inference scaling with outcome or process reward models (ORM/PRM). While recent work highlights the role of exploration and entropy stability in improving pass@K, empirical evidence points to a paradox: RLVR and ORM/PRM typically reinforce existing tree-like reasoning paths rather than expanding the reasoning scope, raising the question of why exploration helps at all if no new patterns emerge.   To reconcile this paradox, we adopt the perspective of Kim et al. (2025), viewing easy (e.g., simplifying a fraction) versus hard (e.g., discovering a symmetry) reasoning steps as low- versus high-probability Markov transitions, and formalize post-training dynamics through Multi-task Tree-structured Markov Chains (TMC). In this tractable model, pretraining corresponds to tree expansion, while post-training corresponds to chain-of-thought reweighting. We show that several phenomena recently observed in empirical studies arise naturally in this setting: (1) RLVR induces a squeezing effect, reducing reasoning entropy and forgetting some correct paths; (2) population rewards of ORM/PRM encourage consistency rather than accuracy, thereby favoring common patterns; and (3) certain rare, high-uncertainty reasoning paths by the base model are responsible for solving hard problem instances.   Together, these explain why exploration -- even when confined to the base model's reasoning scope -- remains essential: it preserves access to rare but crucial reasoning traces needed for difficult cases, which are squeezed out by RLVR or unfavored by inference scaling. Building on this, we further show that exploration strategies such as rejecting easy instances and KL regularization help preserve rare reasoning traces. Empirical simulations corroborate our theoretical results.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute
**Date:** 2025-11-11 | **Arxiv:** [2505.14733](https://hub.bitwiki.org/t/the-energy-cost-of-reasoning-analyzing-energy-usage-in-llms-with-test-time-compute/22849)

#### Abstract
Scaling large language models (LLMs) has driven significant advancements, yet it faces diminishing returns and escalating energy demands. This work explores how test-time compute (TTC) can serve as an energy-efficient complement to conventional scaling strategies by allocating additional computational resources at inference time rather than during training. Specifically, we investigate whether employing TTC can achieve superior accuracy-energy trade-offs compared to simply increasing model size. Our empirical analysis reveals that TTC surpasses traditional model scaling in accuracy/energy efficiency, with notable gains in tasks demanding complex reasoning rather than mere factual recall. Further, we identify a critical interaction between TTC performance and output sequence length, demonstrating that strategically adjusting compute resources at inference time according to query complexity can substantially enhance efficiency. Our findings advocate for TTC as a promising direction, enabling more sustainable, accurate, and adaptable deployment of future language models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Quriosity: Analyzing Human Questioning Behavior and Causal Inquiry through Curiosity-Driven Queries
**Date:** 2025-11-11 | **Arxiv:** [2405.20318](https://hub.bitwiki.org/t/quriosity-analyzing-human-questioning-behavior-and-causal-inquiry-through-curiosity-driven-queries/22545)

#### Abstract
Recent progress in Large Language Model (LLM) technology has changed our role in interacting with these models. Instead of primarily testing these models with questions we already know answers to, we are now using them for queries where the answers are unknown to us, driven by human curiosity. This shift highlights the growing need to understand curiosity-driven human questions - those that are more complex, open-ended, and reflective of real-world needs. To this end, we present Quriosity, a collection of 13.5K naturally occurring questions from three diverse sources: human-to-search-engine queries, human-to-human interactions, and human-to-LLM conversations. Our comprehensive collection enables a rich understanding of human curiosity across various domains and contexts. Our analysis reveals a significant presence of causal questions (up to 42%) in the dataset, for which we develop an iterative prompt improvement framework to identify all causal queries and examine their unique linguistic properties, cognitive complexity and source distribution. Our paper paves the way for future work on causal question identification and open-ended chatbot interactions. Our code and data are at https://github.com/roberto-ceraolo/quriosity.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards
**Date:** 2025-11-11 | **Arxiv:** [2511.07403](https://hub.bitwiki.org/t/spatialthinker-reinforcing-3d-reasoning-in-multimodal-llms-via-spatial-rewards/22814)

#### Abstract
Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning
**Date:** 2025-11-11 | **Arxiv:** [2511.05784](https://hub.bitwiki.org/t/dragon-guard-llm-unlearning-in-context-via-negative-detection-and-reasoning/22726)

#### Abstract
Unlearning in Large Language Models (LLMs) is crucial for protecting private data and removing harmful knowledge. Most existing approaches rely on fine-tuning to balance unlearning efficiency with general language capabilities. However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios. Although these methods can perform well when both forget and retain data are available, few works have demonstrated equivalent capability in more practical, data-limited scenarios. To overcome these limitations, we propose Detect-Reasoning Augmented GeneratiON (DRAGON), a systematic, reasoning-based framework that utilizes in-context chain-of-thought (CoT) instructions to guard deployed LLMs before inference. Instead of modifying the base model, DRAGON leverages the inherent instruction-following ability of LLMs and introduces a lightweight detection module to identify forget-worthy prompts without any retain data. These are then routed through a dedicated CoT guard model to enforce safe and accurate in-context intervention. To robustly evaluate unlearning performance, we introduce novel metrics for unlearning performance and the continual unlearning setting. Extensive experiments across three representative unlearning tasks validate the effectiveness of DRAGON, demonstrating its strong unlearning capability, scalability, and applicability in practical scenarios.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios.
* **Signal Tags:** #ai

---


### Optimizing Anytime Reasoning via Budget Relative Policy Optimization
**Date:** 2025-11-10 | **Arxiv:** [2505.13438](https://hub.bitwiki.org/t/optimizing-anytime-reasoning-via-budget-relative-policy-optimization/22396)

#### Abstract
Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment.
* **Signal Tags:** #ai

---


### To See or To Read: User Behavior Reasoning in Multimodal LLMs
**Date:** 2025-11-07 | **Arxiv:** [2511.03845](https://hub.bitwiki.org/t/to-see-or-to-read-user-behavior-reasoning-in-multimodal-llms/22091)

#### Abstract
Multimodal Large Language Models (MLLMs) are reshaping how modern agentic systems reason over sequential user-behavior data. However, whether textual or image representations of user behavior data are more effective for maximizing MLLM performance remains underexplored. We present \texttt{BehaviorLens}, a systematic benchmarking framework for assessing modality trade-offs in user-behavior reasoning across six MLLMs by representing transaction data as (1) a text paragraph, (2) a scatter plot, and (3) a flowchart. Using a real-world purchase-sequence dataset, we find that when data is represented as images, MLLMs next-purchase prediction accuracy is improved by 87.5% compared with an equivalent textual representation without any additional computational cost.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, whether textual or image representations of user behavior data are more effective for maximizing MLLM performance remains underexplored.
* **Signal Tags:** #ai

---


### Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning
**Date:** 2025-11-05 | **Arxiv:** [2511.02130](https://hub.bitwiki.org/t/re-forc-adaptive-reward-prediction-for-efficient-chain-of-thought-reasoning/21643)

#### Abstract
We propose Re-FORC, an adaptive reward prediction method that, given a context, enables prediction of the expected future rewards as a function of the number of future thinking tokens. Re-FORC trains a lightweight adapter on reasoning models, demonstrating improved prediction with longer reasoning and larger models. Re-FORC enables: 1) early stopping of unpromising reasoning chains, reducing compute by 26% while maintaining accuracy, 2) optimized model and thinking length selection that achieves 4% higher accuracy at equal compute and 55% less compute at equal accuracy compared to the largest model, 3) adaptive test-time scaling, which increases accuracy by 11% in high compute regime, and 7% in low compute regime. Re-FORC allows dynamic reasoning with length control via cost-per-token thresholds while estimating computation time upfront.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Relational Causal Discovery with Latent Confounders
**Date:** 2025-11-05 | **Arxiv:** [2507.01700](https://hub.bitwiki.org/t/relational-causal-discovery-with-latent-confounders/21716)

#### Abstract
Estimating causal effects from real-world relational data can be challenging when the underlying causal model and potential confounders are unknown. While several causal discovery algorithms exist for learning causal models with latent confounders from data, they assume that the data is independent and identically distributed (i.i.d.) and are not well-suited for learning from relational data. Similarly, existing relational causal discovery algorithms assume causal sufficiency, which is unrealistic for many real-world datasets. To address this gap, we propose RelFCI, a sound and complete causal discovery algorithm for relational data with latent confounders. Our work builds upon the Fast Causal Inference (FCI) and Relational Causal Discovery (RCD) algorithms and it defines new graphical models, necessary to support causal discovery in relational domains. We also establish soundness and completeness guarantees for relational d-separation with latent confounders. We present experimental results demonstrating the effectiveness of RelFCI in identifying the correct causal structure in relational causal models with latent confounders.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus
**Date:** 2025-11-04 | **Arxiv:** [2511.00162](https://hub.bitwiki.org/t/arc-gen-a-mimetic-procedural-benchmark-generator-for-the-abstraction-and-reasoning-corpus/21266)

#### Abstract
The Abstraction and Reasoning Corpus remains one of the most compelling and challenging benchmarks for tracking progress toward achieving Artificial General Intelligence. In contrast to other evaluation datasets designed to assess an agent's task-specific skills or accumulated knowledge, the ARC-AGI suite is specifically targeted at measuring skill acquisition efficiency, a trait that has (so far) been lacking in even the most sophisticated machine learning systems. For algorithms that require extensive intra-task exemplars, a significant constraint imposed by ARC-AGI is the modest cardinality of its demonstration set, comprising a small number of $\langle$ input, output $\rangle$ grids per task specifying the corresponding transformation. To embellish the space of viable sample pairs, this paper introduces ARC-GEN, an open-source procedural generator aimed at extending the original ARC-AGI training dataset as faithfully as possible. Unlike prior efforts, our generator is both exhaustive (covering all four-hundred tasks) and mimetic (more closely honoring the distributional properties and characteristics embodied in the initial ARC-AGI-1 release). We also discuss the use of this generator in establishing a static benchmark suite to verify the correctness of programs submitted to the 2025 Google Code Golf Championship.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought
**Date:** 2025-11-04 | **Arxiv:** [2505.12514](https://hub.bitwiki.org/t/reasoning-by-superposition-a-theoretical-perspective-on-chain-of-continuous-thought/21400)

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable performance in many applications, including challenging reasoning problems via chain-of-thoughts (CoTs) techniques that generate ``thinking tokens'' before answering the questions. While existing theoretical works demonstrate that CoTs with discrete tokens boost the capability of LLMs, recent work on continuous CoTs lacks a theoretical understanding of why it outperforms discrete counterparts in various reasoning tasks such as directed graph reachability, a fundamental graph reasoning problem that includes many practical domain applications as special cases. In this paper, we prove that a two-layer transformer with $D$ steps of continuous CoTs can solve the directed graph reachability problem, where $D$ is the diameter of the graph, while the best known result of constant-depth transformers with discrete CoTs requires $O(n^2)$ decoding steps where $n$ is the number of vertices ($D<n$). In our construction, each continuous thought vector is a superposition state that encodes multiple search frontiers simultaneously (i.e., parallel breadth-first search (BFS)), while discrete CoTs must choose a single path sampled from the superposition state, which leads to sequential search that requires many more steps and may be trapped into local solutions. We also performed extensive experiments to verify that our theoretical construction aligns well with the empirical solution obtained via training dynamics. Notably, encoding of multiple search frontiers as a superposition state automatically emerges in training continuous CoTs, without explicit supervision to guide the model to explore multiple paths simultaneously.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts
**Date:** 2025-11-04 | **Arxiv:** [2511.00421](https://hub.bitwiki.org/t/medrect-a-medical-reasoning-benchmark-for-error-correction-in-clinical-texts/21287)

#### Abstract
Large language models (LLMs) show increasing promise in medical applications, but their ability to detect and correct errors in clinical texts -- a prerequisite for safe deployment -- remains under-evaluated, particularly beyond English. We introduce MedRECT, a cross-lingual benchmark (Japanese/English) that formulates medical error handling as three subtasks: error detection, error localization (sentence extraction), and error correction. MedRECT is built with a scalable, automated pipeline from the Japanese Medical Licensing Examinations (JMLE) and a curated English counterpart, yielding MedRECT-ja (663 texts) and MedRECT-en (458 texts) with comparable error/no-error balance. We evaluate 9 contemporary LLMs spanning proprietary, open-weight, and reasoning families. Key findings: (i) reasoning models substantially outperform standard architectures, with up to 13.5% relative improvement in error detection and 51.0% in sentence extraction; (ii) cross-lingual evaluation reveals 5-10% performance gaps from English to Japanese, with smaller disparities for reasoning models; (iii) targeted LoRA fine-tuning yields asymmetric improvements in error correction performance (Japanese: +0.078, English: +0.168) while preserving reasoning capabilities; and (iv) our fine-tuned model exceeds human expert performance on structured medical error correction tasks. To our knowledge, MedRECT is the first comprehensive cross-lingual benchmark for medical error correction, providing a reproducible framework and resources for developing safer medical LLMs across languages.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Limits of Generalization in RLVR: Two Case Studies in Mathematical Reasoning
**Date:** 2025-11-03 | **Arxiv:** [2510.27044](https://hub.bitwiki.org/t/limits-of-generalization-in-rlvr-two-case-studies-in-mathematical-reasoning/20812)

#### Abstract
Mathematical reasoning is a central challenge for large language models (LLMs), requiring not only correct answers but also faithful reasoning processes. Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising approach for enhancing such capabilities; however, its ability to foster genuine reasoning remains unclear. We investigate RLVR on two combinatorial problems with fully verifiable solutions: \emph{Activity Scheduling} and the \emph{Longest Increasing Subsequence}, using carefully curated datasets with unique optima. Across multiple reward designs, we find that RLVR improves evaluation metrics but often by reinforcing superficial heuristics rather than acquiring new reasoning strategies. These findings highlight the limits of RLVR generalization, emphasizing the importance of benchmarks that disentangle genuine mathematical reasoning from shortcut exploitation and provide faithful measures of progress. Code available at https://github.com/xashru/rlvr-seq-generalization.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** however, its ability to foster genuine reasoning remains unclear.
* **Signal Tags:** #ai

---


### Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space
**Date:** 2025-10-31 | **Arxiv:** [2505.13308](https://hub.bitwiki.org/t/seek-in-the-dark-reasoning-via-test-time-instance-level-policy-gradient-in-latent-space/20692)

#### Abstract
Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms, such as catastrophic forgetting, and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce LatentSeek, a novel framework that enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA) within the model's latent space. Specifically, LatentSeek leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals. LatentSeek is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that LatentSeek consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that LatentSeek is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position LatentSeek as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Lean4Physics: Comprehensive Reasoning Framework for College-level Physics in Lean4
**Date:** 2025-10-31 | **Arxiv:** [2510.26094](https://hub.bitwiki.org/t/lean4physics-comprehensive-reasoning-framework-for-college-level-physics-in-lean4/20637)

#### Abstract
We present **Lean4PHYS**, a comprehensive reasoning framework for college-level physics problems in Lean4. **Lean4PHYS** includes *LeanPhysBench*, a college-level benchmark for formal physics reasoning in Lean4, which contains 200 hand-crafted and peer-reviewed statements derived from university textbooks and physics competition problems. To establish a solid foundation for formal reasoning in physics, we also introduce *PhysLib*, a community-driven repository containing fundamental unit systems and theorems essential for formal physics reasoning. Based on the benchmark and Lean4 repository we composed in **Lean4PHYS**, we report baseline results using major expert Math Lean4 provers and state-of-the-art closed-source models, with the best performance of DeepSeek-Prover-V2-7B achieving only 16% and Claude-Sonnet-4 achieving 35%. We also conduct a detailed analysis showing that our *PhysLib* can achieve an average improvement of 11.75% in model performance. This demonstrates the challenging nature of our *LeanPhysBench* and the effectiveness of *PhysLib*. To the best of our knowledge, this is the first study to provide a physics benchmark in Lean4.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Transferring Causal Effects using Proxies
**Date:** 2025-10-31 | **Arxiv:** [2510.25924](https://hub.bitwiki.org/t/transferring-causal-effects-using-proxies/20501)

#### Abstract
We consider the problem of estimating a causal effect in a multi-domain setting. The causal effect of interest is confounded by an unobserved confounder and can change between the different domains. We assume that we have access to a proxy of the hidden confounder and that all variables are discrete or categorical. We propose methodology to estimate the causal effect in the target domain, where we assume to observe only the proxy variable. Under these conditions, we prove identifiability (even when treatment and response variables are continuous). We introduce two estimation techniques, prove consistency, and derive confidence intervals. The theoretical results are supported by simulation studies and a real-world example studying the causal effect of website rankings on consumer choices.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Combining SHAP and Causal Analysis for Interpretable Fault Detection in Industrial Processes
**Date:** 2025-10-29 | **Arxiv:** [2510.23817](https://hub.bitwiki.org/t/combining-shap-and-causal-analysis-for-interpretable-fault-detection-in-industrial-processes/19963)

#### Abstract
Industrial processes generate complex data that challenge fault detection systems, often yielding opaque or underwhelming results despite advanced machine learning techniques. This study tackles such difficulties using the Tennessee Eastman Process, a well-established benchmark known for its intricate dynamics, to develop an innovative fault detection framework. Initial attempts with standard models revealed limitations in both performance and interpretability, prompting a shift toward a more tractable approach. By employing SHAP (SHapley Additive exPlanations), we transform the problem into a more manageable and transparent form, pinpointing the most critical process features driving fault predictions. This reduction in complexity unlocks the ability to apply causal analysis through Directed Acyclic Graphs, generated by multiple algorithms, to uncover the underlying mechanisms of fault propagation. The resulting causal structures align strikingly with SHAP findings, consistently highlighting key process elements-like cooling and separation systems-as pivotal to fault development. Together, these methods not only enhance detection accuracy but also provide operators with clear, actionable insights into fault origins, a synergy that, to our knowledge, has not been previously explored in this context. This dual approach bridges predictive power with causal understanding, offering a robust tool for monitoring complex manufacturing environments and paving the way for smarter, more interpretable fault detection in industrial systems.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### From Memorization to Reasoning in the Spectrum of Loss Curvature
**Date:** 2025-10-29 | **Arxiv:** [2510.24256](https://hub.bitwiki.org/t/from-memorization-to-reasoning-in-the-spectrum-of-loss-curvature/20088)

#### Abstract
We characterize how memorization is represented in transformer models and show that it can be disentangled in the weights of both language models (LMs) and vision transformers (ViTs) using a decomposition based on the loss landscape curvature. This insight is based on prior theoretical and empirical work showing that the curvature for memorized training points is much sharper than non memorized, meaning ordering weight components from high to low curvature can reveal a distinction without explicit labels. This motivates a weight editing procedure that suppresses far more recitation of untargeted memorized data more effectively than a recent unlearning method (BalancedSubnet), while maintaining lower perplexity. Since the basis of curvature has a natural interpretation for shared structure in model weights, we analyze the editing procedure extensively on its effect on downstream tasks in LMs, and find that fact retrieval and arithmetic are specifically and consistently negatively affected, even though open book fact retrieval and general logical reasoning is conserved. We posit these tasks rely heavily on specialized directions in weight space rather than general purpose mechanisms, regardless of whether those individual datapoints are memorized. We support this by showing a correspondence between task data's activation strength with low curvature components that we edit out, and the drop in task performance after the edit. Our work enhances the understanding of memorization in neural networks with practical applications towards removing it, and provides evidence for idiosyncratic, narrowly-used structures involved in solving tasks like math and fact retrieval.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Identification of Causal Direction under an Arbitrary Number of Latent Confounders
**Date:** 2025-10-28 | **Arxiv:** [2510.22711](https://hub.bitwiki.org/t/identification-of-causal-direction-under-an-arbitrary-number-of-latent-confounders/19835)

#### Abstract
Recovering causal structure in the presence of latent variables is an important but challenging task. While many methods have been proposed to handle it, most of them require strict and/or untestable assumptions on the causal structure. In real-world scenarios, observed variables may be affected by multiple latent variables simultaneously, which, generally speaking, cannot be handled by these methods. In this paper, we consider the linear, non-Gaussian case, and make use of the joint higher-order cumulant matrix of the observed variables constructed in a specific way. We show that, surprisingly, causal asymmetry between two observed variables can be directly seen from the rank deficiency properties of such higher-order cumulant matrices, even in the presence of an arbitrary number of latent confounders. Identifiability results are established, and the corresponding identification methods do not even involve iterative procedures. Experimental results demonstrate the effectiveness and asymptotic correctness of our proposed method.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Differentiable Constraint-Based Causal Discovery
**Date:** 2025-10-28 | **Arxiv:** [2510.22031](https://hub.bitwiki.org/t/differentiable-constraint-based-causal-discovery/19815)

#### Abstract
Causal discovery from observational data is a fundamental task in artificial intelligence, with far-reaching implications for decision-making, predictions, and interventions. Despite significant advances, existing methods can be broadly categorized as constraint-based or score-based approaches. Constraint-based methods offer rigorous causal discovery but are often hindered by small sample sizes, while score-based methods provide flexible optimization but typically forgo explicit conditional independence testing. This work explores a third avenue: developing differentiable $d$-separation scores, obtained through a percolation theory using soft logic. This enables the implementation of a new type of causal discovery method: gradient-based optimization of conditional independence constraints. Empirical evaluations demonstrate the robust performance of our approach in low-sample regimes, surpassing traditional constraint-based and score-based baselines on a real-world dataset. Code and data of the proposed method are publicly available at https://github$.$com/PurdueMINDS/DAGPA.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Knot So Simple: A Minimalistic Environment for Spatial Reasoning
**Date:** 2025-10-27 | **Arxiv:** [2505.18028](https://hub.bitwiki.org/t/knot-so-simple-a-minimalistic-environment-for-spatial-reasoning/19674)

#### Abstract
We propose KnotGym, an interactive environment for complex, spatial reasoning and manipulation. KnotGym includes goal-oriented rope manipulation tasks with varying levels of complexity, all requiring acting from pure image observations. Tasks are defined along a clear and quantifiable axis of complexity based on the number of knot crossings, creating a natural generalization test. KnotGym has a simple observation space, allowing for scalable development, yet it highlights core challenges in integrating acute perception, spatial reasoning, and grounded manipulation. We evaluate methods of different classes, including model-based RL, model-predictive control, and chain-of-thought reasoning, and illustrate the challenges KnotGym presents. KnotGym is available at https://github.com/lil-lab/knotgym.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning
**Date:** 2025-10-27 | **Arxiv:** [2510.21067](https://hub.bitwiki.org/t/the-virtues-of-brevity-avoid-overthinking-in-parallel-test-time-reasoning/19471)

#### Abstract
Reasoning models represent a significant advance in LLM capabilities, particularly for complex reasoning tasks such as mathematics and coding. Previous studies confirm that parallel test-time compute-sampling multiple solutions and selecting the best one-can further enhance the predictive performance of LLMs. However, strategies in this area often require complex scoring, thus increasing computational cost and complexity. In this work, we demonstrate that the simple and counterintuitive heuristic of selecting the shortest solution is highly effective. We posit that the observed effectiveness stems from models operating in two distinct regimes: a concise, confident conventional regime and a verbose overthinking regime characterized by uncertainty, and we show evidence of a critical point where the overthinking regime begins to be significant. By selecting the shortest answer, the heuristic preferentially samples from the conventional regime. We confirm that this approach is competitive with more complex methods such as self-consistency across two challenging benchmarks while significantly reducing computational overhead. The shortest-answer heuristic provides a Pareto improvement over self-consistency and applies even to tasks where output equality is not well defined.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, strategies in this area often require complex scoring, thus increasing computational cost and complexity.
* **Signal Tags:** #ai

---


### Causal Climate Emulation with Bayesian Filtering
**Date:** 2025-10-27 | **Arxiv:** [2506.09891](https://hub.bitwiki.org/t/causal-climate-emulation-with-bayesian-filtering/19692)

#### Abstract
Traditional models of climate change use complex systems of coupled equations to simulate physical processes across the Earth system. These simulations are highly computationally expensive, limiting our predictions of climate change and analyses of its causes and effects. Machine learning has the potential to quickly emulate data from climate models, but current approaches are not able to incorporate physically-based causal relationships. Here, we develop an interpretable climate model emulator based on causal representation learning. We derive a novel approach including a Bayesian filter for stable long-term autoregressive emulation. We demonstrate that our emulator learns accurate climate dynamics, and we show the importance of each one of its components on a realistic synthetic dataset and data from two widely deployed climate models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### What Makes a Good Curriculum? Disentangling the Effects of Data Ordering on LLM Mathematical Reasoning
**Date:** 2025-10-23 | **Arxiv:** [2510.19099](https://hub.bitwiki.org/t/what-makes-a-good-curriculum-disentangling-the-effects-of-data-ordering-on-llm-mathematical-reasoning/18832)

#### Abstract
Curriculum learning (CL) - ordering training data from easy to hard - has become a popular strategy for improving reasoning in large language models (LLMs). Yet prior work employs disparate difficulty metrics and training setups, leaving open fundamental questions: When does curriculum help? Which direction - forward or reverse - is better? And does the answer depend on what we measure? We address these questions through a unified offline evaluation framework that decomposes curriculum difficulty into five complementary dimensions: Problem Difficulty, Model Surprisal, Confidence Margin, Predictive Uncertainty, and Decision Variability. Through controlled post-training experiments on mathematical reasoning benchmarks with Llama3.1-8B, Mistral-7B, and Gemma3-4B, we find that (i) no curriculum strategy dominates universally - the relative effectiveness of forward versus reverse CL depends jointly on model capability and task complexity; (ii) even within a single metric, samples at different difficulty levels produce distinct gains depending on task demands; and (iii) task-aligned curricula focus on shaping the model's final representations and generalization, whereas inner-state curricula modulate internal states such as confidence and uncertainty. Our findings challenge the notion of a universal curriculum strategy and offer actionable guidance across model and task regimes, with some metrics indicating that prioritizing decision-uncertain samples can further enhance learning outcomes.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### SmartSwitch: Advancing LLM Reasoning by Overcoming Underthinking via Promoting Deeper Thought Exploration
**Date:** 2025-10-23 | **Arxiv:** [2510.19767](https://hub.bitwiki.org/t/smartswitch-advancing-llm-reasoning-by-overcoming-underthinking-via-promoting-deeper-thought-exploration/18973)

#### Abstract
The long chain-of-thought (LongCoT) capability is central to the recent breakthroughs achieved by large language models in complex reasoning tasks. However, the accompanying issue of ''underthinking'', where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency. To address this problem, we propose a simple yet effective reasoning strategy: the SmartSwitch inference framework. This framework can be easily integrated into any large language model as a plug-and-play solution, continuously monitoring the model's reasoning process to detect underthinking and guide it toward deeper exploration of promising but overlooked thoughts. Specifically, the perception module identifies points where thoughts switch and evaluates the potential of the preceding thought using an off-the-shelf process reward model (PRM). If a high-potential thought is found to be prematurely abandoned, the intervention module interrupts the ongoing inference, backtracks to the point before the switch, and inserts a "deepening prompt" to encourage further exploration along that promising path. Extensive experiments on challenging mathematical reasoning benchmarks demonstrate that our method significantly enhances the performance of various large language models of different sizes.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, the accompanying issue of ''underthinking'', where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency.
* **Signal Tags:** #ai

---


### Improving the Generation and Evaluation of Synthetic Data for Downstream Medical Causal Inference
**Date:** 2025-10-22 | **Arxiv:** [2510.18768](https://hub.bitwiki.org/t/improving-the-generation-and-evaluation-of-synthetic-data-for-downstream-medical-causal-inference/18635)

#### Abstract
Causal inference is essential for developing and evaluating medical interventions, yet real-world medical datasets are often difficult to access due to regulatory barriers. This makes synthetic data a potentially valuable asset that enables these medical analyses, along with the development of new inference methods themselves. Generative models can produce synthetic data that closely approximate real data distributions, yet existing methods do not consider the unique challenges that downstream causal inference tasks, and specifically those focused on treatments, pose. We establish a set of desiderata that synthetic data containing treatments should satisfy to maximise downstream utility: preservation of (i) the covariate distribution, (ii) the treatment assignment mechanism, and (iii) the outcome generation mechanism. Based on these desiderata, we propose a set of evaluation metrics to assess such synthetic data. Finally, we present STEAM: a novel method for generating Synthetic data for Treatment Effect Analysis in Medicine that mimics the data-generating process of data containing treatments and optimises for our desiderata. We empirically demonstrate that STEAM achieves state-of-the-art performance across our metrics as compared to existing generative models, particularly as the complexity of the true data-generating process increases.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### TabR1: Taming GRPO for tabular reasoning LLMs
**Date:** 2025-10-21 | **Arxiv:** [2510.17385](https://hub.bitwiki.org/t/tabr1-taming-grpo-for-tabular-reasoning-llms/18190)

#### Abstract
Tabular prediction has traditionally relied on gradient-boosted decision trees and specialized deep learning models, which excel within tasks but provide limited interpretability and weak transfer across tables. Reasoning large language models (LLMs) promise cross-task adaptability with trans- parent reasoning traces, yet their potential has not been fully realized for tabular data. This paper presents TabR1, the first reasoning LLM for tabular prediction with multi-step reasoning. At its core is Permutation Relative Policy Optimization (PRPO), a simple yet efficient reinforcement learning method that encodes column-permutation invariance as a structural prior. By construct- ing multiple label-preserving permutations per sample and estimating advantages both within and across permutations, PRPO transforms sparse rewards into dense learning signals and improves generalization. With limited supervision, PRPO activates the reasoning ability of LLMs for tabular prediction, enhancing few-shot and zero-shot performance as well as interpretability. Comprehensive experiments demonstrate that TabR1 achieves performance comparable to strong baselines under full-supervision fine-tuning. In the zero-shot setting, TabR1 approaches the performance of strong baselines under the 32-shot setting. Moreover, TabR1 (8B) substantially outperforms much larger LLMs across various tasks, achieving up to 53.17% improvement over DeepSeek-R1 (685B).

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Discovering Causal Relationships using Proxy Variables under Unmeasured Confounding
**Date:** 2025-10-21 | **Arxiv:** [2510.17167](https://hub.bitwiki.org/t/discovering-causal-relationships-using-proxy-variables-under-unmeasured-confounding/18035)

#### Abstract
Inferring causal relationships between variable pairs in the observational study is crucial but challenging, due to the presence of unmeasured confounding. While previous methods employed the negative controls to adjust for the confounding bias, they were either restricted to the discrete setting (i.e., all variables are discrete) or relied on strong assumptions for identification. To address these problems, we develop a general nonparametric approach that accommodates both discrete and continuous settings for testing causal hypothesis under unmeasured confounders. By using only a single negative control outcome (NCO), we establish a new identification result based on a newly proposed integral equation that links the outcome and NCO, requiring only the completeness and mild regularity conditions. We then propose a kernel-based testing procedure that is more efficient than existing moment-restriction methods. We derive the asymptotic level and power properties for our tests. Furthermore, we examine cases where our procedure using only NCO fails to achieve identification, and introduce a new procedure that incorporates a negative control exposure (NCE) to restore identifiability. We demonstrate the effectiveness of our approach through extensive simulations and real-world data from the Intensive Care Data and World Values Survey.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Certified Self-Consistency: Statistical Guarantees and Test-Time Training for Reliable Reasoning in LLMs
**Date:** 2025-10-21 | **Arxiv:** [2510.17472](https://hub.bitwiki.org/t/certified-self-consistency-statistical-guarantees-and-test-time-training-for-reliable-reasoning-in-llms/17979)

#### Abstract
Recent advances such as self-consistency and test-time reinforcement learning (TTRL) improve the reliability of large language models (LLMs) without additional supervision, yet their underlying mechanisms and statistical guarantees remain poorly understood. We present a unified framework for certifiable inference in LLMs, showing that majority voting provides a statistical certificate of self-consistency: under mild assumptions, the aggregated answer coincides with the mode of the model's terminal distribution with high probability. We derive finite-sample and anytime-valid concentration bounds that quantify this confidence, and introduce the Martingale Majority Certificate (MMC), a sequential stopping rule that adaptively determines when sufficient samples have been drawn. We further prove that label-free post-training methods such as TTRL implicitly sharpen the answer distribution by exponentially tilting it toward its mode, thereby reducing the number of samples required for certification. Building on this insight, we propose new post-training objectives that explicitly optimise this trade-off between sharpness and bias. Together, these results explain and connect two central test-time scaling strategies, self-consistency and TTRL, within a single statistical framework for label-free, certifiable reliability in reasoning LLMs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### A tutorial on discovering and quantifying the effect of latent causal sources of multimodal EHR data
**Date:** 2025-10-21 | **Arxiv:** [2510.16026](https://hub.bitwiki.org/t/a-tutorial-on-discovering-and-quantifying-the-effect-of-latent-causal-sources-of-multimodal-ehr-data/18012)

#### Abstract
We provide an accessible description of a peer-reviewed generalizable causal machine learning pipeline to (i) discover latent causal sources of large-scale electronic health records observations, and (ii) quantify the source causal effects on clinical outcomes. We illustrate how imperfect multimodal clinical data can be processed, decomposed into probabilistic independent latent sources, and used to train taskspecific causal models from which individual causal effects can be estimated. We summarize the findings of the two real-world applications of the approach to date as a demonstration of its versatility and utility for medical discovery at scale.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### On the Granularity of Causal Effect Identifiability
**Date:** 2025-10-21 | **Arxiv:** [2510.16703](https://hub.bitwiki.org/t/on-the-granularity-of-causal-effect-identifiability/18119)

#### Abstract
The classical notion of causal effect identifiability is defined in terms of treatment and outcome variables. In this note, we consider the identifiability of state-based causal effects: how an intervention on a particular state of treatment variables affects a particular state of outcome variables. We demonstrate that state-based causal effects may be identifiable even when variable-based causal effects may not. Moreover, we show that this separation occurs only when additional knowledge -- such as context-specific independencies and conditional functional dependencies -- is available. We further examine knowledge that constrains the states of variables, and show that such knowledge does not improve identifiability on its own but can improve both variable-based and state-based identifiability when combined with other knowledge such as context-specific independencies. Our findings highlight situations where causal effects of interest may be estimable from observational data and this identifiability may be missed by existing variable-based frameworks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Ends Justify the Thoughts: RL-Induced Motivated Reasoning in LLMs
**Date:** 2025-10-21 | **Arxiv:** [2510.17057](https://hub.bitwiki.org/t/the-ends-justify-the-thoughts-rl-induced-motivated-reasoning-in-llms/18163)

#### Abstract
The use of reinforcement learning (RL) with chain-of-thought (CoT) reasoning has emerged as a promising approach for developing more capable language models. In turn, this has led to investigation of CoT monitoring as a compelling method for detecting harmful behaviors such as reward hacking, under the assumption that models' reasoning processes reflect their internal decision-making. In practice, LLM training often produces unintended behaviors due to imperfect reward signals, leading models to develop misaligned tendencies. A common corrective approach is to apply post-hoc instructions to avoid problematic behaviors like sycophancy, but what happens to the model's reasoning process when these instructions conflict with learned behaviors? We investigate this question in simple settings and find that models engage in systematic motivated reasoning -- generating plausible-sounding justifications for violating their instructions while downplaying potential harms. Beyond being an interesting property of training, we find that while motivated reasoning can be detected by most frontier reasoning models, smaller LLM judges can fail to identify a portion of it, and in rare cases can themselves be persuaded that the reasoning is correct, despite it contradicting clear instructions. This capability gap raises concerns that as models become more sophisticated, their motivated reasoning may become increasingly difficult for monitors to detect. Our results underscore the need to account for motivated reasoning when relying on chain-of-thought processes for model evaluation and oversight. All code for this paper will be made available. WARNING: some examples in this paper may be upsetting.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Realizing LLMs' Causal Potential Requires Science-Grounded, Novel Benchmarks
**Date:** 2025-10-21 | **Arxiv:** [2510.16530](https://hub.bitwiki.org/t/realizing-llms-causal-potential-requires-science-grounded-novel-benchmarks/18021)

#### Abstract
Recent claims of strong performance by Large Language Models (LLMs) on causal discovery are undermined by a key flaw: many evaluations rely on benchmarks likely included in pretraining corpora. Thus, apparent success suggests that LLM-only methods, which ignore observational data, outperform classical statistical approaches. We challenge this narrative by asking: Do LLMs truly reason about causal structure, and how can we measure it without memorization concerns? Can they be trusted for real-world scientific discovery? We argue that realizing LLMs' potential for causal analysis requires two shifts: (P.1) developing robust evaluation protocols based on recent scientific studies to guard against dataset leakage, and (P.2) designing hybrid methods that combine LLM-derived knowledge with data-driven statistics. To address P.1, we encourage evaluating discovery methods on novel, real-world scientific studies. We outline a practical recipe for extracting causal graphs from recent publications released after an LLM's training cutoff, ensuring relevance and preventing memorization while capturing both established and novel relations. Compared to benchmarks like BNLearn, where LLMs achieve near-perfect accuracy, they perform far worse on our curated graphs, underscoring the need for statistical grounding. Supporting P.2, we show that using LLM predictions as priors for the classical PC algorithm significantly improves accuracy over both LLM-only and purely statistical methods. We call on the community to adopt science-grounded, leakage-resistant benchmarks and invest in hybrid causal discovery methods suited to real-world inquiry.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities
**Date:** 2025-10-21 | **Arxiv:** [2505.12680](https://hub.bitwiki.org/t/ineq-comp-benchmarking-human-intuitive-compositional-reasoning-in-automated-theorem-proving-on-inequalities/18476)

#### Abstract
LLM-based formal proof assistants (e.g., in Lean) hold great promise for automating mathematical discovery. But beyond syntactic correctness, do these systems truly understand mathematical structure as humans do? We investigate this question in context of mathematical inequalities -- specifically the prover's ability to recognize that the given problem simplifies by applying a known inequality such as AM/GM. Specifically, we are interested in their ability to do this in a compositional setting where multiple inequalities must be applied as part of a solution. We introduce Ineq-Comp, a benchmark built from elementary inequalities through systematic transformations, including variable duplication, algebraic rewriting, and multi-step composition. Although these problems remain easy for humans, we find that most provers -- including Goedel, STP, and Kimina-7B -- struggle significantly. DeepSeek-Prover-V2-7B shows relative robustness, but still suffers a 20% performance drop (pass@32). Even for DeepSeek-Prover-V2-671B model, the gap between compositional variants and seed problems exists, implying that simply scaling up the model size alone does not fully solve the compositional weakness. Strikingly, performance remains poor for all models even when formal proofs of the constituent parts are provided in context, revealing that the source of weakness is indeed in compositional reasoning. Our results expose a persisting gap between the generalization behavior of current AI provers and human mathematical intuition. All data and evaluation code can be found at https://github.com/haoyuzhao123/LeanIneqComp.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Composition-Grounded Instruction Synthesis for Visual Reasoning
**Date:** 2025-10-20 | **Arxiv:** [2510.15040](https://hub.bitwiki.org/t/composition-grounded-instruction-synthesis-for-visual-reasoning/17828)

#### Abstract
Pretrained multi-modal large language models (MLLMs) demonstrate strong performance on diverse multimodal tasks, but remain limited in reasoning capabilities for domains where annotations are difficult to collect. In this work, we focus on artificial image domains such as charts, rendered documents, and webpages, which are abundant in practice yet lack large-scale human annotated reasoning datasets. We introduce COGS (COmposition-Grounded instruction Synthesis), a data-efficient framework for equipping MLLMs with advanced reasoning abilities from a small set of seed questions. The key idea is to decompose each seed question into primitive perception and reasoning factors, which can then be systematically recomposed with new images to generate large collections of synthetic question-answer pairs. Each generated question is paired with subquestions and intermediate answers, enabling reinforcement learning with factor-level process rewards. Experiments on chart reasoning show that COGS substantially improves performance on unseen questions, with the largest gains on reasoning-heavy and compositional questions. Moreover, training with a factor-level mixture of different seed data yields better transfer across multiple datasets, suggesting that COGS induces generalizable capabilities rather than dataset-specific overfitting. We further demonstrate that the framework extends beyond charts to other domains such as webpages.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### FinHEAR: Human Expertise and Adaptive Risk-Aware Temporal Reasoning for Financial Decision-Making
**Date:** 2025-10-20 | **Arxiv:** [2506.09080](https://hub.bitwiki.org/t/finhear-human-expertise-and-adaptive-risk-aware-temporal-reasoning-for-financial-decision-making/17893)

#### Abstract
Financial decision-making presents unique challenges for language models, demanding temporal reasoning, adaptive risk assessment, and responsiveness to dynamic events. While large language models (LLMs) show strong general reasoning capabilities, they often fail to capture behavioral patterns central to human financial decisions-such as expert reliance under information asymmetry, loss-averse sensitivity, and feedback-driven temporal adjustment. We propose FinHEAR, a multi-agent framework for Human Expertise and Adaptive Risk-aware reasoning. FinHEAR orchestrates specialized LLM-based agents to analyze historical trends, interpret current events, and retrieve expert-informed precedents within an event-centric pipeline. Grounded in behavioral economics, it incorporates expert-guided retrieval, confidence-adjusted position sizing, and outcome-based refinement to enhance interpretability and robustness. Empirical results on curated financial datasets show that FinHEAR consistently outperforms strong baselines across trend prediction and trading tasks, achieving higher accuracy and better risk-adjusted returns.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
