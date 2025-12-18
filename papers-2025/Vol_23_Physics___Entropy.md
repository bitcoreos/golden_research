# Vol 23 Physics   Entropy
*Enriched by BITCOREOS | Phase 4 Batch 5*

---

### Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-Constrained RL
**Date:** 2025-11-18 | **Arxiv:** [2511.11592](https://arxiv.org/abs/2511.11592)

#### Abstract
Maximum entropy has become a mainstream off-policy reinforcement learning (RL) framework for balancing exploitation and exploration. However, two bottlenecks still limit further performance improvement: (1) non-stationary Q-value estimation caused by jointly injecting entropy and updating its weighting parameter, i.e., temperature; and (2) short-sighted local entropy tuning that adjusts temperature only according to the current single-step entropy, without considering the effect of cumulative entropy over time. In this paper, we extends maximum entropy framework by proposing a trajectory entropy-constrained reinforcement learning (TECRL) framework to address these two challenges. Within this framework, we first separately learn two Q-functions, one associated with reward and the other with entropy, ensuring clean and stable value targets unaffected by temperature updates. Then, the dedicated entropy Q-function, explicitly quantifying the expected cumulative entropy, enables us to enforce a trajectory entropy constraint and consequently control the policy long-term stochasticity. Building on this TECRL framework, we develop a practical off-policy algorithm, DSAC-E, by extending the state-of-the-art distributional soft actor-critic with three refinements (DSAC-T). Empirical results on the OpenAI Gym benchmark demonstrate that our DSAC-E can achieve higher returns and better stability.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, two bottlenecks still limit further performance improvement: (1) non-stationary Q-value estimation caused by jointly injecting entropy and updating its weighting parameter, i.
* **Signal Tags:** #ai

---


### Learning Time-Scale Invariant Population-Level Neural Representations
**Date:** 2025-11-18 | **Arxiv:** [2511.13022](https://arxiv.org/abs/2511.13022)

#### Abstract
General-purpose foundation models for neural time series can help accelerate neuroscientific discoveries and enable applications such as brain computer interfaces (BCIs). A key component in scaling these models is population-level representation learning, which leverages information across channels to capture spatial as well as temporal structure. Population-level approaches have recently shown that such representations can be both efficient to learn on top of pretrained temporal encoders and produce useful representations for decoding a variety of downstream tasks. However, these models remain sensitive to mismatches in preprocessing, particularly on time-scales, between pretraining and downstream settings. We systematically examine how time-scale mismatches affects generalization and find that existing representations lack invariance. To address this, we introduce Time-scale Augmented Pretraining (TSAP), which consistently improves robustness to different time-scales across decoding tasks and builds invariance in the representation space. These results highlight handling preprocessing diversity as a key step toward building generalizable neural foundation models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, these models remain sensitive to mismatches in preprocessing, particularly on time-scales, between pretraining and downstream settings.
* **Signal Tags:** #ai

---


### Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning
**Date:** 2025-12-08 | **Arxiv:** [2512.05591](https://arxiv.org/abs/2512.05591)

#### Abstract
Large language model post-training relies on reinforcement learning to improve model capability and alignment quality. However, the off-policy training paradigm introduces distribution shift, which often pushes the policy beyond the trust region, leading to training instabilities manifested as fluctuations in policy entropy and unstable gradients. Although PPO-Clip mitigates this issue through importance clipping, it still overlooks the global distributional shift of actions. To address these challenges, we propose using the entropy ratio between the current and previous policies as a new global metric that effectively quantifies the relative change in policy exploration throughout updates. Building on this metric, we introduce an \textbf{Entropy Ratio Clipping} (ERC) mechanism that imposes bidirectional constraints on the entropy ratio. This stabilizes policy updates at the global distribution level and compensates for the inability of PPO-clip to regulate probability shifts of un-sampled actions. We integrate ERC into both DAPO and GPPO reinforcement learning algorithms. Experiments across multiple benchmarks show that ERC consistently improves performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, the off-policy training paradigm introduces distribution shift, which often pushes the policy beyond the trust region, leading to training instabilities manifested as fluctuations in policy entropy and unstable gradients.
* **Signal Tags:** #ai

---


### CLIP: Client-Side Invariant Pruning for Mitigating Stragglers in Secure Federated Learning
**Date:** 2025-10-21 | **Arxiv:** [2510.16694](https://arxiv.org/abs/2510.16694)

#### Abstract
Secure federated learning (FL) preserves data privacy during distributed model training. However, deploying such frameworks across heterogeneous devices results in performance bottlenecks, due to straggler clients with limited computational or network capabilities, slowing training for all participating clients. This paper introduces the first straggler mitigation technique for secure aggregation with deep neural networks. We propose CLIP, a client-side invariant neuron pruning technique coupled with network-aware pruning, that addresses compute and network bottlenecks due to stragglers during training with minimal accuracy loss. Our technique accelerates secure FL training by 13% to 34% across multiple datasets (CIFAR10, Shakespeare, FEMNIST) with an accuracy impact of between 1.3% improvement to 2.6% reduction.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, deploying such frameworks across heterogeneous devices results in performance bottlenecks, due to straggler clients with limited computational or network capabilities, slowing training for all participating clients.
* **Signal Tags:** #ai

---


### CrossRF: A Domain-Invariant Deep Learning Approach for RF Fingerprinting
**Date:** 2025-10-21 | **Arxiv:** [2505.18200](https://arxiv.org/abs/2505.18200)

#### Abstract
Radio Frequency (RF) fingerprinting offers a promising approach for drone identification and security, although it suffers from significant performance degradation when operating on different transmission channels. This paper presents CrossRF, a domain-invariant deep learning approach that addresses the problem of cross-channel RF fingerprinting for Unmanned Aerial Vehicle (UAV) identification. Our approach aims to minimize the domain gap between different RF channels by using adversarial learning to train a more robust model that maintains consistent identification performance despite channel variations. We validate our approach using the UAVSig dataset, comprising real-world over-the-air RF signals from identical drone models operating across several frequency channels, ensuring that the findings correspond to real-world scenarios. The experimental results show CrossRF's efficiency, achieving up to 99.03% accuracy when adapting from Channel 3 to Channel 4, compared to only 26.39% using conventional methods. The model maintains robust performance in more difficult multi-channel scenarios (87.57% accuracy adapting from Channels 1,3 to 2,4) and achieves 89.45% accuracy with 0.9 precision for controller classification. These results confirm CrossRF's ability to significantly reduce performance degradation due to cross-channel variations while maintaining high identification accuracy with minimal training data requirements, making it particularly suitable for practical drone security applications.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning
**Date:** 2025-10-14 | **Arxiv:** [2510.10959](https://arxiv.org/abs/2510.10959)

#### Abstract
Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance.
* **Signal Tags:** #ai

---


### Arbitrary Entropy Policy Optimization Breaks The Exploration Bottleneck of Reinforcement Learning
**Date:** 2025-10-10 | **Arxiv:** [2510.08141](https://arxiv.org/abs/2510.08141)

#### Abstract
Reinforcement Learning (RL) is essential for enhancing the reasoning capabilities of large language models (LLMs), yet the widely adopted Group Relative Policy Optimization (GRPO) suffers from entropy collapse, causing exploration to vanish and policies to converge prematurely. As a result, RL is widely believed to be incapable of expanding the reasoning frontier of LLMs. Existing entropy-regularized methods introduce an inevitable trade-off between reward and entropy, leading to exploration accompanied by non-negligible optimization bias. In this work, we prove that temperature-guided REINFORCE can modulate policy entropy, and propose Arbitrary Entropy Policy Optimization (AEPO), which reformulates entropy regularization as a policy-gradient optimization problem. Rather than manipulating entropy directly, AEPO implicitly regulates it by applying a REINFORCE regularization term on temperature-adjusted samples, ensuring that entropy is controlled but never dominates optimization, thereby enabling arbitrary and principled entropy regulation. Experiments show that AEPO outperforms RL baselines on both pass@1 and pass@$k$, and even surpasses the base model on pass@1024. By modulating entropy precisely, AEPO achieves more effective optimization dynamics and provides direct empirical evidence that entropy, exploration, and performance are intrinsically linked.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning
**Date:** 2025-10-10 | **Arxiv:** [2510.08526](https://arxiv.org/abs/2510.08526)

#### Abstract
In the pursuit of finding an optimal policy, reinforcement learning (RL) methods generally ignore the properties of learned policies apart from their expected return. Thus, even when successful, it is difficult to characterize which policies will be learned and what they will do. In this work, we present a theoretical framework for policy optimization that guarantees convergence to a particular optimal policy, via vanishing entropy regularization and a temperature decoupling gambit. Our approach realizes an interpretable, diversity-preserving optimal policy as the regularization temperature vanishes and ensures the convergence of policy derived objects--value functions and return distributions. In a particular instance of our method, for example, the realized policy samples all optimal actions uniformly. Leveraging our temperature decoupling gambit, we present an algorithm that estimates, to arbitrary accuracy, the return distribution associated to its interpretable, diversity-preserving optimal policy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Neural Network Characterization and Entropy Regulated Data Balancing through Principal Component Analysis
**Date:** 2025-10-02 | **Arxiv:** [2312.01392](https://arxiv.org/abs/2312.01392)

#### Abstract
This paper examines in detail the geometric structure of principal component analysis (PCA) by considering in detail the distributions of both unrotated and rotated MNIST digits in the space defined by the lowest order PCA components. Since digits possessing salient geometric features are mapped to restricted regions far from the origin, they are predicted by neural networks with a greater accuracy than digits that are mapped to broad, diffuse and overlapping volumes of the low order PCA space. Motivated by these results, a new quantity, the local PCA entropy, obtained by dividing the spatial region spanned by the low order principal components into histogram bins and evaluating the entropy associated with the number of occurrences of each input class within a bin, is introduced. The metric locates the input data records that yield the largest confusion in prediction accuracy within reduced coordinate volumes that optimally discriminate among geometric features. As an example of the potential utility of the local PCA entropy, a simple data balancing procedure is realized by oversampling the data records in regions of large local entropy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning
**Date:** 2025-09-26 | **Arxiv:** [2509.20712](https://arxiv.org/abs/2509.20712)

#### Abstract
Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}oordinating \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Information Entropy-Based Scheduling for Communication-Efficient Decentralized Learning
**Date:** 2025-09-16 | **Arxiv:** [2507.17426](https://arxiv.org/abs/2507.17426)

#### Abstract
This paper addresses decentralized stochastic gradient descent (D-SGD) over resource-constrained networks by introducing node-based and link-based scheduling strategies to enhance communication efficiency. In each iteration of the D-SGD algorithm, only a few disjoint subsets of nodes or links are randomly activated, subject to a given communication cost constraint. We propose a novel importance metric based on information entropy to determine node and link scheduling probabilities. We validate the effectiveness of our approach through extensive simulations, comparing it against state-of-the-art methods, including betweenness centrality (BC) for node scheduling and \textit{MATCHA} for link scheduling. The results show that our method consistently outperforms the BC-based method in the node scheduling case, achieving faster convergence with up to 60\% lower communication budgets. At higher communication budgets (above 60\%), our method maintains comparable or superior performance. In the link scheduling case, our method delivers results that are superior to or on par with those of \textit{MATCHA}.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations
**Date:** 2025-12-08 | **Arxiv:** [2512.05156](https://arxiv.org/abs/2512.05156)

#### Abstract
Evaluating faithfulness of Large Language Models (LLMs) to a given task is a complex challenge. We propose two new unsupervised metrics for faithfulness evaluation using insights from information theory and thermodynamics. Our approach treats an LLM as a bipartite information engine where hidden layers act as a Maxwell demon controlling transformations of context $C $ into answer $A$ via prompt $Q$. We model Question-Context-Answer (QCA) triplets as probability distributions over shared topics. Topic transformations from $C$ to $Q$ and $A$ are modeled as transition matrices ${\bf Q}$ and ${\bf A}$ encoding the query goal and actual result, respectively. Our semantic faithfulness (SF) metric quantifies faithfulness for any given QCA triplet by the Kullback-Leibler (KL) divergence between these matrices. Both matrices are inferred simultaneously via convex optimization of this KL divergence, and the final SF metric is obtained by mapping the minimal divergence onto the unit interval [0,1], where higher scores indicate greater faithfulness. Furthermore, we propose a thermodynamics-based semantic entropy production (SEP) metric in answer generation, and show that high faithfulness generally implies low entropy production. The SF and SEP metrics can be used jointly or separately for LLM evaluation and hallucination control. We demonstrate our framework on LLM summarization of corporate SEC 10-K filings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Contrastive Entropy Bounds for Density and Conditional Density Decomposition
**Date:** 2025-11-18 | **Arxiv:** [2511.12903](https://arxiv.org/abs/2511.12903)

#### Abstract
This paper studies the interpretability of neural network features from a Bayesian Gaussian view, where optimizing a cost is reaching a probabilistic bound; learning a model approximates a density that makes the bound tight and the cost optimal, often with a Gaussian mixture density. The two examples are Mixture Density Networks (MDNs) using the bound for the marginal and autoencoders using the conditional bound. It is a known result, not only for autoencoders, that minimizing the error between inputs and outputs maximizes the dependence between inputs and the middle.   We use Hilbert space and decomposition to address cases where a multiple-output network produces multiple centers defining a Gaussian mixture. Our first finding is that an autoencoder's objective is equivalent to maximizing the trace of a Gaussian operator, the sum of eigenvalues under bases orthonormal w.r.t. the data and model distributions. This suggests that, when a one-to-one correspondence as needed in autoencoders is unnecessary, we can instead maximize the nuclear norm of this operator, the sum of singular values, to maximize overall rank rather than trace. Thus the trace of a Gaussian operator can be used to train autoencoders, and its nuclear norm can be used as divergence to train MDNs.   Our second test uses inner products and norms in a Hilbert space to define bounds and costs. Such bounds often have an extra norm compared to KL-based bounds, which increases sample diversity and prevents the trivial solution where a multiple-output network produces the same constant, at the cost of requiring a sample batch to estimate and optimize. We propose an encoder-mixture-decoder architecture whose decoder is multiple-output, producing multiple centers per sample, potentially tightening the bound. Assuming the data are small-variance Gaussian mixtures, this upper bound can be tracked and analyzed quantitatively.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Privacy-Preserving Explainable AIoT Application via SHAP Entropy Regularization
**Date:** 2025-11-14 | **Arxiv:** [2511.09775](https://arxiv.org/abs/2511.09775)

#### Abstract
The widespread integration of Artificial Intelligence of Things (AIoT) in smart home environments has amplified the demand for transparent and interpretable machine learning models. To foster user trust and comply with emerging regulatory frameworks, the Explainable AI (XAI) methods, particularly post-hoc techniques such as SHapley Additive exPlanations (SHAP), and Local Interpretable Model-Agnostic Explanations (LIME), are widely employed to elucidate model behavior. However, recent studies have shown that these explanation methods can inadvertently expose sensitive user attributes and behavioral patterns, thereby introducing new privacy risks. To address these concerns, we propose a novel privacy-preserving approach based on SHAP entropy regularization to mitigate privacy leakage in explainable AIoT applications. Our method incorporates an entropy-based regularization objective that penalizes low-entropy SHAP attribution distributions during training, promoting a more uniform spread of feature contributions. To evaluate the effectiveness of our approach, we developed a suite of SHAP-based privacy attacks that strategically leverage model explanation outputs to infer sensitive information. We validate our method through comparative evaluations using these attacks alongside utility metrics on benchmark smart home energy consumption datasets. Experimental results demonstrate that SHAP entropy regularization substantially reduces privacy leakage compared to baseline models, while maintaining high predictive accuracy and faithful explanation fidelity. This work contributes to the development of privacy-preserving explainable AI techniques for secure and trustworthy AIoT applications.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, recent studies have shown that these explanation methods can inadvertently expose sensitive user attributes and behavioral patterns, thereby introducing new privacy risks.
* **Signal Tags:** #ai

---


### Beyond MSE: Ordinal Cross-Entropy for Probabilistic Time Series Forecasting
**Date:** 2025-11-14 | **Arxiv:** [2511.10200](https://arxiv.org/abs/2511.10200)

#### Abstract
Time series forecasting is an important task that involves analyzing temporal dependencies and underlying patterns (such as trends, cyclicality, and seasonality) in historical data to predict future values or trends. Current deep learning-based forecasting models primarily employ Mean Squared Error (MSE) loss functions for regression modeling. Despite enabling direct value prediction, this method offers no uncertainty estimation and exhibits poor outlier robustness. To address these limitations, we propose OCE-TS, a novel ordinal classification approach for time series forecasting that replaces MSE with Ordinal Cross-Entropy (OCE) loss, preserving prediction order while quantifying uncertainty through probability output. Specifically, OCE-TS begins by discretizing observed values into ordered intervals and deriving their probabilities via a parametric distribution as supervision signals. Using a simple linear model, we then predict probability distributions for each timestep. The OCE loss is computed between the cumulative distributions of predicted and ground-truth probabilities, explicitly preserving ordinal relationships among forecasted values. Through theoretical analysis using influence functions, we establish that cross-entropy (CE) loss exhibits superior stability and outlier robustness compared to MSE loss. Empirically, we compared OCE-TS with five baseline models-Autoformer, DLinear, iTransformer, TimeXer, and TimeBridge-on seven public time series datasets. Using MSE and Mean Absolute Error (MAE) as evaluation metrics, the results demonstrate that OCE-TS consistently outperforms benchmark models. The codeis publicly available at: https://github.com/Shi-hm/OCE-TS.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training
**Date:** 2025-11-14 | **Arxiv:** [2511.10333](https://arxiv.org/abs/2511.10333)

#### Abstract
Training large language models (LLMs) poses significant challenges regarding computational resources and memory capacity. Although distributed training techniques help mitigate these issues, they still suffer from considerable communication overhead. Existing approaches primarily rely on static gradient compression to enhance communication efficiency; however, these methods neglect the dynamic nature of evolving gradients during training, leading to performance degradation. Accelerating LLM training via compression without sacrificing performance remains a challenge. In this paper, we propose an entropy-driven dynamic gradient compression framework called EDGC. The core concept is to adjust the compression rate during LLM training based on the evolving trends of gradient entropy, taking into account both compression efficiency and error. EDGC consists of three key components.First, it employs a down-sampling method to efficiently estimate gradient entropy, reducing computation overhead. Second, it establishes a theoretical model linking compression rate with gradient entropy, enabling more informed compression decisions. Lastly, a window-based adjustment mechanism dynamically adapts the compression rate across pipeline stages, improving communication efficiency and maintaining model performance. We implemented EDGC on a 32-NVIDIA-V100 cluster and a 64-NVIDIA-H100 cluster to train GPT2-2.5B and GPT2-12.1B, respectively. The results show that EDGC significantly reduces communication latency and training time by up to 46.45% and 16.13% while preserving LLM accuracy.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** however, these methods neglect the dynamic nature of evolving gradients during training, leading to performance degradation.
* **Signal Tags:** #ai

---


### From Exploration to Exploitation: A Two-Stage Entropy RLVR Approach for Noise-Tolerant MLLM Training
**Date:** 2025-11-12 | **Arxiv:** [2511.07738](https://arxiv.org/abs/2511.07738)

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) for Multimodal Large Language Models (MLLMs) is highly dependent on high-quality labeled data, which is often scarce and prone to substantial annotation noise in real-world scenarios. Existing unsupervised RLVR methods, including pure entropy minimization, can overfit to incorrect labels and limit the crucial reward ranking signal for Group-Relative Policy Optimization (GRPO). To address these challenges and enhance noise tolerance, we propose a novel two-stage, token-level entropy optimization method for RLVR. This approach dynamically guides the model from exploration to exploitation during training. In the initial exploration phase, token-level entropy maximization promotes diverse and stochastic output generation, serving as a strong regularizer that prevents premature convergence to noisy labels and ensures sufficient intra-group variation, which enables more reliable reward gradient estimation in GRPO. As training progresses, the method transitions into the exploitation phase, where token-level entropy minimization encourages the model to produce confident and deterministic outputs, thereby consolidating acquired knowledge and refining prediction accuracy. Empirically, across three MLLM backbones - Qwen2-VL-2B, Qwen2-VL-7B, and Qwen2.5-VL-3B - spanning diverse noise settings and multiple tasks, our phased strategy consistently outperforms prior approaches by unifying and enhancing external, internal, and entropy-based methods, delivering robust and superior performance across the board.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Recursive Dynamics in Fast-Weights Homeostatic Reentry Networks: Toward Reflective Intelligence
**Date:** 2025-11-11 | **Arxiv:** [2511.06798](https://arxiv.org/abs/2511.06798)

#### Abstract
This study introduces the Fast-Weights Homeostatic Reentry Layer (FH-RL), a neural mechanism that integrates fast-weight associative memory, homeostatic regularization, and learned reentrant feedback to approximate self-referential computation in neural networks. Unlike standard transformer architectures that operate in a purely feedforward manner during inference, FH-RL enables internal recurrence without external looping, allowing prior latent states to be dynamically re-entered into the ongoing computation stream. We conduct controlled experiments sweeping the reentry gain $γ$ and evaluate emergent internal dynamics using three novel metrics: the Information Reentry Ratio (IRR), Eigen-Spectrum Recursion Index (ESRI), and Representational Drift Periodicity (RDP). Results show that reentry quantity increases proportionally with~$γ$, while the learned feedback matrix $W_r$ remains bounded and becomes more structured at moderate gains. Critically, a stable reflective band emerges around $γ\approx 0.10-0.20$, where internal feedback is maximally expressive yet spectrally stable: IRR rises smoothly, ESRI remains near zero, and RDP exhibits consistent low-frequency cycles. These findings provide quantitative evidence that reflective, thought-like internal processing can arise from a principled balance between feedback amplification and homeostatic regulation, linking modern fast-weight architectures to theories of cortical reentry and recursive cognition.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Blind Inverse Game Theory: Jointly Decoding Rewards and Rationality in Entropy-Regularized Competitive Games
**Date:** 2025-11-11 | **Arxiv:** [2511.05640](https://arxiv.org/abs/2511.05640)

#### Abstract
Inverse Game Theory (IGT) methods based on the entropy-regularized Quantal Response Equilibrium (QRE) offer a tractable approach for competitive settings, but critically assume the agents' rationality parameter (temperature $τ$) is known a priori. When $τ$ is unknown, a fundamental scale ambiguity emerges that couples $τ$ with the reward parameters ($θ$), making them statistically unidentifiable. We introduce Blind-IGT, the first statistical framework to jointly recover both $θ$ and $τ$ from observed behavior. We analyze this bilinear inverse problem and establish necessary and sufficient conditions for unique identification by introducing a normalization constraint that resolves the scale ambiguity. We propose an efficient Normalized Least Squares (NLS) estimator and prove it achieves the optimal $\mathcal{O}(N^{-1/2})$ convergence rate for joint parameter recovery. When strong identifiability conditions fail, we provide partial identification guarantees through confidence set construction. We extend our framework to Markov games and demonstrate optimal convergence rates with strong empirical performance even when transition dynamics are unknown.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Impacts of Individual Fairness on Group Fairness from the Perspective of Generalized Entropy
**Date:** 2025-11-11 | **Arxiv:** [2202.11966](https://arxiv.org/abs/2202.11966)

#### Abstract
This paper investigates how the degree of group fairness changes when the degree of individual fairness is actively controlled. As a metric quantifying individual fairness, we consider generalized entropy (GE) recently introduced into machine learning community. To control the degree of individual fairness, we design a classification algorithm satisfying a given degree of individual fairness through an empirical risk minimization (ERM) with a fairness constraint specified in terms of GE. We show the PAC learnability of the fair ERM problem by proving that the true fairness degree does not deviate much from an empirical one with high probability for finite VC dimension if the sample size is big enough. Our experiments show that strengthening individual fairness degree does not always lead to enhancement of group fairness.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Which Similarity-Sensitive Entropy?
**Date:** 2025-11-07 | **Arxiv:** [2511.03849](https://arxiv.org/abs/2511.03849)

#### Abstract
A canonical step in quantifying a system is to measure its entropy. Shannon entropy and other traditional entropy measures capture only the information encoded in the frequencies of a system's elements. Recently, Leinster, Cobbold, and Reeve (LCR) introduced a method that also captures the rich information encoded in the similarities and differences among elements, yielding similarity-sensitive entropy. More recently, the Vendi score (VS) was introduced as an alternative, raising the question of how LCR and VS compare, and which is preferable. Here we address these questions conceptually, analytically, and experimentally, using 53 machine-learning datasets. We show that LCR and VS can differ by orders of magnitude and can capture complementary information about a system, except in limiting cases. We demonstrate that both LCR and VS depend on how similarities are scaled and introduce the concept of ``half distance'' to parameterize this dependence. We prove that VS provides an upper bound on LCR for several values of the Rényi-Hill order parameter and conjecture that this bound holds for all values. We conclude that VS is preferable only when interpreting elements as linear combinations of a more fundamental set of ``ur-elements'' or when the system or dataset possesses a quantum-mechanical character. In the broader circumstance where one seeks simply to capture the rich information encoded by similarity, LCR is favored; nevertheless, for certain half-distances the two methods can complement each other.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Decoupled Entropy Minimization
**Date:** 2025-11-06 | **Arxiv:** [2511.03256](https://arxiv.org/abs/2511.03256)

#### Abstract
Entropy Minimization (EM) is beneficial to reducing class overlap, bridging domain gap, and restricting uncertainty for various tasks in machine learning, yet its potential is limited. To study the internal mechanism of EM, we reformulate and decouple the classical EM into two parts with opposite effects: cluster aggregation driving factor (CADF) rewards dominant classes and prompts a peaked output distribution, while gradient mitigation calibrator (GMC) penalizes high-confidence classes based on predicted probabilities. Furthermore, we reveal the limitations of classical EM caused by its coupled formulation: 1) reward collapse impedes the contribution of high-certainty samples in the learning process, and 2) easy-class bias induces misalignment between output distribution and label distribution. To address these issues, we propose Adaptive Decoupled Entropy Minimization (AdaDEM), which normalizes the reward brought from CADF and employs a marginal entropy calibrator (MEC) to replace GMC. AdaDEM outperforms DEM*, an upper-bound variant of classical EM, and achieves superior performance across various imperfectly supervised learning tasks in noisy and dynamic environments.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### SPIKE: Stable Physics-Informed Kernel Evolution Method for Solving Hyperbolic Conservation Laws
**Date:** 2025-10-22 | **Arxiv:** [2510.18266](https://arxiv.org/abs/2510.18266)

#### Abstract
We introduce the Stable Physics-Informed Kernel Evolution (SPIKE) method for numerical computation of inviscid hyperbolic conservation laws. SPIKE resolves a fundamental paradox: how strong-form residual minimization can capture weak solutions containing discontinuities. SPIKE employs reproducing kernel representations with regularized parameter evolution, where Tikhonov regularization provides a smooth transition mechanism through shock formation, allowing the dynamics to traverse shock singularities. This approach automatically maintains conservation, tracks characteristics, and captures shocks satisfying Rankine-Hugoniot conditions within a unified framework requiring no explicit shock detection or artificial viscosity. Numerical validation across scalar and vector-valued conservation laws confirms the method's effectiveness.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Principle of Uncertain Maximum Entropy
**Date:** 2025-10-17 | **Arxiv:** [2305.09868](https://arxiv.org/abs/2305.09868)

#### Abstract
The Principle of Maximum Entropy is a rigorous technique for estimating an unknown distribution given partial information while simultaneously minimizing bias. However, an important requirement for applying the principle is that the available information be provided error-free (Jaynes 1982). We relax this requirement using a memoryless communication channel as a framework to derive a new, more general principle. We show our new principle provides an upper bound on the entropy of the unknown distribution and the amount of information lost due to the use of a given communications channel is unknown unless the unknown distribution's entropy is also known. Using our new principle we provide a new interpretation of the classic principle and experimentally show its performance relative to the classic principle and other generally applicable solutions. Finally, we present a simple algorithm for solving our new principle and an approximation useful when samples are limited.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, an important requirement for applying the principle is that the available information be provided error-free (Jaynes 1982).
* **Signal Tags:** #ai

---


### Deciphering Invariant Feature Decoupling in Source-free Time Series Forecasting with Proxy Denoising
**Date:** 2025-10-08 | **Arxiv:** [2510.05589](https://arxiv.org/abs/2510.05589)

#### Abstract
The proliferation of mobile devices generates a massive volume of time series across various domains, where effective time series forecasting enables a variety of real-world applications. This study focuses on a new problem of source-free domain adaptation for time series forecasting. It aims to adapt a pretrained model from sufficient source time series to the sparse target time series domain without access to the source data, embracing data protection regulations. To achieve this, we propose TimePD, the first source-free time series forecasting framework with proxy denoising, where large language models (LLMs) are employed to benefit from their generalization capabilities. Specifically, TimePD consists of three key components: (1) dual-branch invariant disentangled feature learning that enforces representation- and gradient-wise invariance by means of season-trend decomposition; (2) lightweight, parameter-free proxy denoising that dynamically calibrates systematic biases of LLMs; and (3) knowledge distillation that bidirectionally aligns the denoised prediction and the original target prediction. Extensive experiments on real-world datasets offer insight into the effectiveness of the proposed TimePD, outperforming SOTA baselines by 9.3% on average.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### What Scales in Cross-Entropy Scaling Law?
**Date:** 2025-10-07 | **Arxiv:** [2510.04067](https://arxiv.org/abs/2510.04067)

#### Abstract
The cross-entropy scaling law has long served as a key tool for guiding the development of large language models. It shows that cross-entropy loss decreases in a predictable power-law rate as the model size increases. However, recent evidence indicates that this law breaks down at very large scales: the loss decreases more slowly than expected, which causes significant trouble for developing large language models. In this paper, we hypothesize that the root cause lies in the fact that cross-entropy itself does not truly scale; instead, only one of its hidden components does. To investigate this, we introduce a novel decomposition of cross-entropy into three parts: Error-Entropy, Self-Alignment, and Confidence. We show both theoretically and empirically that this decomposition precisely captures the training dynamics and optimization objectives. Through extensive experiments on multiple datasets and 32 models spanning five orders of magnitude in size, we find that only error-entropy follows a robust power-law scaling, while the other two terms remain largely invariant. Moreover, error-entropy constitutes the dominant share of cross-entropy in small models but diminishes in proportion as models grow larger. This explains why the cross-entropy scaling law appears accurate at small scales but fails at very large ones. Our findings establish the error-entropy scaling law as a more accurate description of model behavior. We believe it will have wide applications in the training, understanding, and future development of large language models.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, recent evidence indicates that this law breaks down at very large scales: the loss decreases more slowly than expected, which causes significant trouble for developing large language models.
* **Signal Tags:** #ai

---


### Shift-Invariant Attribute Scoring for Kolmogorov-Arnold Networks via Shapley Value
**Date:** 2025-10-03 | **Arxiv:** [2510.01663](https://arxiv.org/abs/2510.01663)

#### Abstract
For many real-world applications, understanding feature-outcome relationships is as crucial as achieving high predictive accuracy. While traditional neural networks excel at prediction, their black-box nature obscures underlying functional relationships. Kolmogorov--Arnold Networks (KANs) address this by employing learnable spline-based activation functions on edges, enabling recovery of symbolic representations while maintaining competitive performance. However, KAN's architecture presents unique challenges for network pruning. Conventional magnitude-based methods become unreliable due to sensitivity to input coordinate shifts. We propose \textbf{ShapKAN}, a pruning framework using Shapley value attribution to assess node importance in a shift-invariant manner. Unlike magnitude-based approaches, ShapKAN quantifies each node's actual contribution, ensuring consistent importance rankings regardless of input parameterization. Extensive experiments on synthetic and real-world datasets demonstrate that ShapKAN preserves true node importance while enabling effective network compression. Our approach improves KAN's interpretability advantages, facilitating deployment in resource-constrained environments.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, KAN's architecture presents unique challenges for network pruning.
* **Signal Tags:** #ai

---


### Approximation of differential entropy in Bayesian optimal experimental design
**Date:** 2025-10-02 | **Arxiv:** [2510.00734](https://arxiv.org/abs/2510.00734)

#### Abstract
Bayesian optimal experimental design provides a principled framework for selecting experimental settings that maximize obtained information. In this work, we focus on estimating the expected information gain in the setting where the differential entropy of the likelihood is either independent of the design or can be evaluated explicitly. This reduces the problem to maximum entropy estimation, alleviating several challenges inherent in expected information gain computation.   Our study is motivated by large-scale inference problems, such as inverse problems, where the computational cost is dominated by expensive likelihood evaluations. We propose a computational approach in which the evidence density is approximated by a Monte Carlo or quasi-Monte Carlo surrogate, while the differential entropy is evaluated using standard methods without additional likelihood evaluations. We prove that this strategy achieves convergence rates that are comparable to, or better than, state-of-the-art methods for full expected information gain estimation, particularly when the cost of entropy evaluation is negligible. Moreover, our approach relies only on mild smoothness of the forward map and avoids stronger technical assumptions required in earlier work. We also present numerical experiments, which confirm our theoretical findings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### A space-decoupling framework for optimization on bounded-rank matrices with orthogonally invariant constraints
**Date:** 2025-10-01 | **Arxiv:** [2501.13830](https://arxiv.org/abs/2501.13830)

#### Abstract
Imposing additional constraints on low-rank optimization has garnered growing interest. However, the geometry of coupled constraints hampers the well-developed low-rank structure and makes the problem intricate. To this end, we propose a space-decoupling framework for optimization on bounded-rank matrices with orthogonally invariant constraints. The "space-decoupling" is reflected in several ways. We show that the tangent cone of coupled constraints is the intersection of tangent cones of each constraint. Moreover, we decouple the intertwined bounded-rank and orthogonally invariant constraints into two spaces, leading to optimization on a smooth manifold. Implementing Riemannian algorithms on this manifold is painless as long as the geometry of additional constraints is known. In addition, we unveil the equivalence between the reformulated problem and the original problem. Numerical experiments on real-world applications -- spherical data fitting, graph similarity measuring, low-rank SDP, model reduction of Markov processes, reinforcement learning, and deep learning -- validate the superiority of the proposed framework.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, the geometry of coupled constraints hampers the well-developed low-rank structure and makes the problem intricate.
* **Signal Tags:** #ai

---


### Class-Invariant Test-Time Augmentation for Domain Generalization
**Date:** 2025-09-19 | **Arxiv:** [2509.14420](https://arxiv.org/abs/2509.14420)

#### Abstract
Deep models often suffer significant performance degradation under distribution shifts. Domain generalization (DG) seeks to mitigate this challenge by enabling models to generalize to unseen domains. Most prior approaches rely on multi-domain training or computationally intensive test-time adaptation. In contrast, we propose a complementary strategy: lightweight test-time augmentation. Specifically, we develop a novel Class-Invariant Test-Time Augmentation (CI-TTA) technique. The idea is to generate multiple variants of each input image through elastic and grid deformations that nevertheless belong to the same class as the original input. Their predictions are aggregated through a confidence-guided filtering scheme that remove unreliable outputs, ensuring the final decision relies on consistent and trustworthy cues. Extensive Experiments on PACS and Office-Home datasets demonstrate consistent gains across different DG algorithms and backbones, highlighting the effectiveness and generality of our approach.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation
**Date:** 2025-08-13 | **Arxiv:** [2508.04946](https://arxiv.org/abs/2508.04946)

#### Abstract
Simultaneous Speech Translation (SimulST) systems stream in audio while simultaneously emitting translated text or speech. Such systems face the significant challenge of balancing translation quality and latency. We introduce a strategy to optimize this tradeoff: wait for more input only if you gain information by doing so. Based on this strategy, we present Regularized Entropy INformation Adaptation (REINA), a novel loss to train an adaptive policy using an existing non-streaming translation model. We derive REINA from information theory principles and show that REINA helps push the reported Pareto frontier of the latency/quality tradeoff over prior works. Utilizing REINA, we train a SimulST model on French, Spanish and German, both from and into English. Training on only open source or synthetically generated data, we achieve state-of-the-art (SOTA) streaming results for models of comparable size. We also introduce a metric for streaming efficiency, quantitatively showing REINA improves the latency/quality trade-off by as much as 21% compared to prior approaches, normalized against non-streaming baseline BLEU scores.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Physics-Informed Machine Learning for Characterizing System Stability
**Date:** 2025-11-13 | **Arxiv:** [2511.08831](https://arxiv.org/abs/2511.08831)

#### Abstract
In the design and operation of complex dynamical systems, it is essential to ensure that all state trajectories of the dynamical system converge to a desired equilibrium within a guaranteed stability region. Yet, for many practical systems -- especially in aerospace -- this region cannot be determined a priori and is often challenging to compute. One of the most common methods for computing the stability region is to identify a Lyapunov function. A Lyapunov function is a positive function whose time derivative along system trajectories is non-positive, which provides a sufficient condition for stability and characterizes an estimated stability region. However, existing methods of characterizing a stability region via a Lyapunov function often rely on explicit knowledge of the system governing equations. In this work, we present a new physics-informed machine learning method of characterizing an estimated stability region by inferring a Lyapunov function from system trajectory data that treats the dynamical system as a black box and does not require explicit knowledge of the system governing equations. In our presented Lyapunov function Inference method (LyapInf), we propose a quadratic form for the unknown Lyapunov function and fit the unknown quadratic operator to system trajectory data by minimizing the average residual of the Zubov equation, a first-order partial differential equation whose solution yields a Lyapunov function. The inferred quadratic Lyapunov function can then characterize an ellipsoidal estimate of the stability region. Numerical results on benchmark examples demonstrate that our physics-informed stability analysis method successfully characterizes a near-maximal ellipsoid of the system stability region associated with the inferred Lyapunov function without requiring knowledge of the system governing equations.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, existing methods of characterizing a stability region via a Lyapunov function often rely on explicit knowledge of the system governing equations.
* **Signal Tags:** #ai

---


### A Rapid Physics-Informed Machine Learning Framework Based on Extreme Learning Machine for Inverse Stefan Problems
**Date:** 2025-10-27 | **Arxiv:** [2510.21426](https://arxiv.org/abs/2510.21426)

#### Abstract
The inverse Stefan problem, as a typical phase-change problem with moving boundaries, finds extensive applications in science and engineering. Recent years have seen the applications of physics-informed neural networks (PINNs) to solving Stefan problems, yet they still exhibit shortcomings in hyperparameter dependency, training efficiency, and prediction accuracy. To address this, this paper develops a physics-informed extreme learning machine (PIELM), a rapid physics-informed learning method framework for inverse Stefan problems. PIELM replaces conventional deep neural networks with an extreme learning machine network. The input weights are fixed in the PIELM framework, and the output weights are determined by optimizing a loss vector of physical laws composed by initial and boundary conditions and governing partial differential equations (PDEs). Then, solving inverse Stefan problems is transformed into finding the Moore-Penrose generalized inverse by the least squares method. Case studies show that the PIELM can increase the prediction accuracy by 3-7 order of magnitude in terms of the relative L2 error, and meanwhile saving more than 94% training time, compared to conventional PINNs.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Neural Thermodynamics: Entropic Forces in Deep and Universal Representation Learning
**Date:** 2025-10-27 | **Arxiv:** [2505.12387](https://arxiv.org/abs/2505.12387)

#### Abstract
With the rapid discovery of emergent phenomena in deep learning and large language models, understanding their cause has become an urgent need. Here, we propose a rigorous entropic-force theory for understanding the learning dynamics of neural networks trained with stochastic gradient descent (SGD) and its variants. Building on the theory of parameter symmetries and an entropic loss landscape, we show that representation learning is crucially governed by emergent entropic forces arising from stochasticity and discrete-time updates. These forces systematically break continuous parameter symmetries and preserve discrete ones, leading to a series of gradient balance phenomena that resemble the equipartition property of thermal systems. These phenomena, in turn, (a) explain the universal alignment of neural representations between AI models and lead to a proof of the Platonic Representation Hypothesis, and (b) reconcile the seemingly contradictory observations of sharpness- and flatness-seeking behavior of deep learning optimization. Our theory and experiments demonstrate that a combination of entropic forces and symmetry breaking is key to understanding emergent phenomena in deep learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### TunnElQNN: A Hybrid Quantum-classical Neural Network for Efficient Learning
**Date:** 2025-10-23 | **Arxiv:** [2505.00933](https://arxiv.org/abs/2505.00933)

#### Abstract
Hybrid quantum-classical neural networks (HQCNNs) represent a promising frontier in machine learning, leveraging the complementary strengths of both models. In this work, we propose the development of TunnElQNN, a non-sequential architecture composed of alternating classical and quantum layers. Within the classical component, we employ the Tunnelling Diode Activation Function (TDAF), inspired by the I-V characteristics of quantum tunnelling. We evaluate the performance of this hybrid model on a synthetic dataset of interleaving half-circle for multi-class classification tasks with varying degrees of class overlap. The model is compared against a baseline hybrid architecture that uses the conventional ReLU activation function (ReLUQNN). Our results show that the TunnElQNN model consistently outperforms the ReLUQNN counterpart. Furthermore, we analyse the decision boundaries generated by TunnElQNN under different levels of class overlap and compare them to those produced by a neural network implementing TDAF within a fully classical architecture. These findings highlight the potential of integrating physics-inspired activation functions with quantum components to enhance the expressiveness and robustness of hybrid quantum-classical machine learning architectures.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learning by Steering the Neural Dynamics: A Statistical Mechanics Perspective
**Date:** 2025-10-16 | **Arxiv:** [2510.11984](https://arxiv.org/abs/2510.11984)

#### Abstract
Despite the striking successes of deep neural networks trained with gradient-based optimization, these methods differ fundamentally from their biological counterparts. This gap raises key questions about how nature achieves robust, sample-efficient learning at minimal energy costs and solves the credit-assignment problem without backpropagation. We take a step toward bridging contemporary AI and computational neuroscience by studying how neural dynamics can support fully local, distributed learning that scales to simple machine-learning benchmarks. Using tools from statistical mechanics, we identify conditions for the emergence of robust dynamical attractors in random asymmetric recurrent networks. We derive a closed-form expression for the number of fixed points as a function of self-coupling strength, and we reveal a phase transition in their structure: below a critical self-coupling, isolated fixed points coexist with exponentially many narrow clusters showing the overlap-gap property; above it, subdominant yet dense and extensive clusters appear. These fixed points become accessible, including to a simple asynchronous dynamical rule, after an algorithm-dependent self-coupling threshold. Building on this analysis, we propose a biologically plausible algorithm for supervised learning with any binary recurrent network. Inputs are mapped to fixed points of the dynamics, by relaxing under transient external stimuli and stabilizing the resulting configurations via local plasticity. We show that our algorithm can learn an entangled version of MNIST, leverages depth to develop hierarchical representations and increase hetero-association capacity, and is applicable to several architectures. Finally, we highlight the strong connection between algorithm performance and the unveiled phase transition, and we suggest a cortex-inspired alternative to self-couplings for its emergence.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought
**Date:** 2025-09-30 | **Arxiv:** [2509.23365](https://arxiv.org/abs/2509.23365)

#### Abstract
Previous work shows that the chain of continuous thought (continuous CoT) improves the reasoning capability of large language models (LLMs) by enabling implicit parallel thinking, and a subsequent work provided theoretical insight by showing that a two-layer transformer equipped with continuous CoT can efficiently solve directed graph reachability by maintaining a superposition of multiple reasoning traces in the continuous thought. However, it remains unclear how the superposition mechanism is naturally learned from gradient-based training methods. To fill this gap, we theoretically analyze the training dynamics of a simplified two-layer transformer on the directed graph reachability problem to unveil how the superposition mechanism emerges during training in two training stages -- (i) a thought-generation stage that autoregressively expands the continuous thought, and (ii) a prediction stage that converts the thought into the final answer. Our analysis reveals that during training using continuous thought, the index-matching logit, an important quantity which reflects the strength of the model's local search ability, will first increase and then remain bounded under mild assumptions. The bounded index-matching logit effectively balances exploration and exploitation during the reasoning process: the model will exploit local problem structures to identify plausible search traces, and assign comparable weights to multiple such traces to explore when it is uncertain about which solution is correct, which results in superposition. Our experimental results tracking the growth of logits further validate our theory.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, it remains unclear how the superposition mechanism is naturally learned from gradient-based training methods.
* **Signal Tags:** #ai

---


### Fast-Forward Lattice Boltzmann: Learning Kinetic Behaviour with Physics-Informed Neural Operators
**Date:** 2025-09-29 | **Arxiv:** [2509.22411](https://arxiv.org/abs/2509.22411)

#### Abstract
The lattice Boltzmann equation (LBE), rooted in kinetic theory, provides a powerful framework for capturing complex flow behaviour by describing the evolution of single-particle distribution functions (PDFs). Despite its success, solving the LBE numerically remains computationally intensive due to strict time-step restrictions imposed by collision kernels. Here, we introduce a physics-informed neural operator framework for the LBE that enables prediction over large time horizons without step-by-step integration, effectively bypassing the need to explicitly solve the collision kernel. We incorporate intrinsic moment-matching constraints of the LBE, along with global equivariance of the full distribution field, enabling the model to capture the complex dynamics of the underlying kinetic system. Our framework is discretization-invariant, enabling models trained on coarse lattices to generalise to finer ones (kinetic super-resolution). In addition, it is agnostic to the specific form of the underlying collision model, which makes it naturally applicable across different kinetic datasets regardless of the governing dynamics. Our results demonstrate robustness across complex flow scenarios, including von Karman vortex shedding, ligament breakup, and bubble adhesion. This establishes a new data-driven pathway for modelling kinetic systems.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Energy-Efficient Hierarchical Neural Network with Fast FPGA-Based Incremental Learning
**Date:** 2025-09-19 | **Arxiv:** [2509.15097](https://arxiv.org/abs/2509.15097)

#### Abstract
The rising computational and energy demands of deep learning, particularly in large-scale architectures such as foundation models and large language models (LLMs), pose significant challenges to sustainability. Traditional gradient-based training methods are inefficient, requiring numerous iterative updates and high power consumption. To address these limitations, we propose a hybrid framework that combines hierarchical decomposition with FPGA-based direct equation solving and incremental learning. Our method divides the neural network into two functional tiers: lower layers are optimized via single-step equation solving on FPGAs for efficient and parallelizable feature extraction, while higher layers employ adaptive incremental learning to support continual updates without full retraining. Building upon this foundation, we introduce the Compound LLM framework, which explicitly deploys LLM modules across both hierarchy levels. The lower-level LLM handles reusable representation learning with minimal energy overhead, while the upper-level LLM performs adaptive decision-making through energy-aware updates. This integrated design enhances scalability, reduces redundant computation, and aligns with the principles of sustainable AI. Theoretical analysis and architectural insights demonstrate that our method reduces computational costs significantly while preserving high model performance, making it well-suited for edge deployment and real-time adaptation in energy-constrained environments.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Expected Free Energy-based Planning as Variational Inference
**Date:** 2025-08-25 | **Arxiv:** [2504.14898](https://arxiv.org/abs/2504.14898)

#### Abstract
We address the problem of planning under uncertainty, where an agent must choose actions that not only achieve desired outcomes but also reduce uncertainty. Traditional methods often treat exploration and exploitation as separate objectives, lacking a unified inferential foundation. Active inference, grounded in the Free Energy Principle, provides such a foundation by minimizing Expected Free Energy (EFE), a cost function that combines utility with epistemic drives, such as ambiguity resolution and novelty seeking. However, the computational burden of EFE minimization had remained a significant obstacle to its scalability. In this paper, we show that EFE-based planning arises naturally from minimizing a variational free energy functional on a generative model augmented with preference and epistemic priors. This result reinforces theoretical consistency with the Free Energy Principle by casting planning under uncertainty itself as a form of variational inference. Our formulation yields policies that jointly support goal achievement and information gain, while incorporating a complexity term that accounts for bounded computational resources. This unifying framework connects and extends existing methods, enabling scalable, resource-aware implementations of active inference agents.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, the computational burden of EFE minimization had remained a significant obstacle to its scalability.
* **Signal Tags:** #ai

---


### Learning to Learn the Macroscopic Fundamental Diagram using Physics-Informed and meta Machine Learning techniques
**Date:** 2025-08-21 | **Arxiv:** [2508.14137](https://arxiv.org/abs/2508.14137)

#### Abstract
The Macroscopic Fundamental Diagram is a popular tool used to describe traffic dynamics in an aggregated way, with applications ranging from traffic control to incident analysis. However, estimating the MFD for a given network requires large numbers of loop detectors, which is not always available in practice. This article proposes a framework harnessing meta-learning, a subcategory of machine learning that trains models to understand and adapt to new tasks on their own, to alleviate the data scarcity challenge. The developed model is trained and tested by leveraging data from multiple cities and exploiting it to model the MFD of other cities with different shares of detectors and topological structures. The proposed meta-learning framework is applied to an ad-hoc Multi-Task Physics-Informed Neural Network, specifically designed to estimate the MFD. Results show an average MSE improvement in flow prediction ranging between ~ 17500 and 36000 (depending on the subset of loop detectors tested). The meta-learning framework thus successfully generalizes across diverse urban settings and improves performance on cities with limited data, demonstrating the potential of using meta-learning when a limited number of detectors is available. Finally, the proposed framework is validated against traditional transfer learning approaches and tested with FitFun, a non-parametric model from the literature, to prove its transferability.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, estimating the MFD for a given network requires large numbers of loop detectors, which is not always available in practice.
* **Signal Tags:** #ai

---
