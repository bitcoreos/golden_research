# Vol 27 Optimization   Theory
*Enriched by BITCOREOS | Phase 4 Batch 6*

---

### ORFit: One-Pass Learning via Bridging Orthogonal Gradient Descent and Recursive Least-Squares
**Date:** 2025-10-22 | **Arxiv:** [2207.13853](https://arxiv.org/abs/2207.13853)

#### Abstract
While large machine learning models have shown remarkable performance in various domains, their training typically requires iterating for many passes over the training data. However, due to computational and memory constraints and potential privacy concerns, storing and accessing all the data is impractical in many real-world scenarios where the data arrives in a stream. In this paper, we investigate the problem of one-pass learning, in which a model is trained on sequentially arriving data without retraining on previous datapoints. Motivated by the demonstrated effectiveness of overparameterized models and the phenomenon of benign overfitting, we propose Orthogonal Recursive Fitting (ORFit), an algorithm for one-pass learning which seeks to perfectly fit each new datapoint while minimally altering the predictions on previous datapoints. ORFit updates the parameters in a direction orthogonal to past gradients, similar to orthogonal gradient descent (OGD) in continual learning. We show that, interestingly, ORFit's update leads to an operation similar to the recursive least-squares (RLS) algorithm in adaptive filtering but with significantly improved memory and computational efficiency, i.e., linear, instead of quadratic, in the number of parameters. To further reduce memory usage, we leverage the structure of the streaming data via an incremental principal component analysis (IPCA). We show that using the principal components is minimax optimal, i.e., it minimizes the worst-case forgetting of previous predictions for unknown future updates. Further, we prove that, for overparameterized linear models, the parameter vector obtained by ORFit matches what the standard multi-pass stochastic gradient descent (SGD) would converge to. Finally, we extend our results to the nonlinear setting for highly overparameterized models, relevant for deep learning.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, due to computational and memory constraints and potential privacy concerns, storing and accessing all the data is impractical in many real-world scenarios where the data arrives in a stream.
* **Signal Tags:** #ai

---


### Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks
**Date:** 2025-11-18 | **Arxiv:** [2511.12985](https://arxiv.org/abs/2511.12985)

#### Abstract
Adversarial examples in neural networks have been extensively studied in Euclidean geometry, but recent advances in \textit{hyperbolic networks} call for a reevaluation of attack strategies in non-Euclidean geometries. Existing methods such as FGSM and PGD apply perturbations without regard to the underlying hyperbolic structure, potentially leading to inefficient or geometrically inconsistent attacks. In this work, we propose a novel adversarial attack that explicitly leverages the geometric properties of hyperbolic space. Specifically, we compute the gradient of the loss function in the tangent space of hyperbolic space, decompose it into a radial (depth) component and an angular (semantic) component, and apply perturbation derived solely from the angular direction. Our method generates adversarial examples by focusing perturbations in semantically sensitive directions encoded in angular movement within the hyperbolic geometry. Empirical results on image classification, cross-modal retrieval tasks and network architectures demonstrate that our attack achieves higher fooling rates than conventional adversarial attacks, while producing high-impact perturbations with deeper insights into vulnerabilities of hyperbolic embeddings. This work highlights the importance of geometry-aware adversarial strategies in curved representation spaces and provides a principled framework for attacking hierarchical embeddings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### The Alignment Game: A Theory of Long-Horizon Alignment Through Recursive Curation
**Date:** 2025-11-18 | **Arxiv:** [2511.12804](https://arxiv.org/abs/2511.12804)

#### Abstract
In self-consuming generative models that train on their own outputs, alignment with user preferences becomes a recursive rather than one-time process. We provide the first formal foundation for analyzing the long-term effects of such recursive retraining on alignment. Under a two-stage curation mechanism based on the Bradley-Terry (BT) model, we model alignment as an interaction between two factions: the Model Owner, who filters which outputs should be learned by the model, and the Public User, who determines which outputs are ultimately shared and retained through interactions with the model. Our analysis reveals three structural convergence regimes depending on the degree of preference alignment: consensus collapse, compromise on shared optima, and asymmetric refinement. We prove a fundamental impossibility theorem: no recursive BT-based curation mechanism can simultaneously preserve diversity, ensure symmetric influence, and eliminate dependence on initialization. Framing the process as dynamic social choice, we show that alignment is not a static goal but an evolving equilibrium, shaped both by power asymmetries and path dependence.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Accelerating Wireless Distributed Learning via Hybrid Split and Federated Learning Optimization
**Date:** 2025-11-26 | **Arxiv:** [2511.19851](https://arxiv.org/abs/2511.19851)

#### Abstract
Federated learning (FL) and split learning (SL) are two effective distributed learning paradigms in wireless networks, enabling collaborative model training across mobile devices without sharing raw data. While FL supports low-latency parallel training, it may converge to less accurate model. In contrast, SL achieves higher accuracy through sequential training but suffers from increased delay. To leverage the advantages of both, hybrid split and federated learning (HSFL) allows some devices to operate in FL mode and others in SL mode. This paper aims to accelerate HSFL by addressing three key questions: 1) How does learning mode selection affect overall learning performance? 2) How does it interact with batch size? 3) How can these hyperparameters be jointly optimized alongside communication and computational resources to reduce overall learning delay? We first analyze convergence, revealing the interplay between learning mode and batch size. Next, we formulate a delay minimization problem and propose a two-stage solution: a block coordinate descent method for a relaxed problem to obtain a locally optimal solution, followed by a rounding algorithm to recover integer batch sizes with near-optimal performance. Experimental results demonstrate that our approach significantly accelerates convergence to the target accuracy compared to existing methods.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Improving Generalization of Neural Combinatorial Optimization for Vehicle Routing Problems via Test-Time Projection Learning
**Date:** 2025-10-31 | **Arxiv:** [2506.02392](https://arxiv.org/abs/2506.02392)

#### Abstract
Neural Combinatorial Optimization (NCO) has emerged as a promising learning-based paradigm for addressing Vehicle Routing Problems (VRPs) by minimizing the need for extensive manual engineering. While existing NCO methods, trained on small-scale instances (e.g., 100 nodes), have demonstrated considerable success on problems of similar scale, their performance significantly degrades when applied to large-scale scenarios. This degradation arises from the distributional shift between training and testing data, rendering policies learned on small instances ineffective for larger problems. To overcome this limitation, we introduce a novel learning framework driven by Large Language Models (LLMs). This framework learns a projection between the training and testing distributions, which is then deployed to enhance the scalability of the NCO model. Notably, unlike prevailing techniques that necessitate joint training with the neural network, our approach operates exclusively during the inference phase, obviating the need for model retraining. Extensive experiments demonstrate that our method enables a backbone model (trained on 100-node instances) to achieve superior performance on large-scale Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) of up to 100K nodes from diverse distributions.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### MILES: Modality-Informed Learning Rate Scheduler for Balancing Multimodal Learning
**Date:** 2025-10-21 | **Arxiv:** [2510.17394](https://arxiv.org/abs/2510.17394)

#### Abstract
The aim of multimodal neural networks is to combine diverse data sources, referred to as modalities, to achieve enhanced performance compared to relying on a single modality. However, training of multimodal networks is typically hindered by modality overfitting, where the network relies excessively on one of the available modalities. This often yields sub-optimal performance, hindering the potential of multimodal learning and resulting in marginal improvements relative to unimodal models. In this work, we present the Modality-Informed Learning ratE Scheduler (MILES) for training multimodal joint fusion models in a balanced manner. MILES leverages the differences in modality-wise conditional utilization rates during training to effectively balance multimodal learning. The learning rate is dynamically adjusted during training to balance the speed of learning from each modality by the multimodal model, aiming for enhanced performance in both multimodal and unimodal predictions. We extensively evaluate MILES on four multimodal joint fusion tasks and compare its performance to seven state-of-the-art baselines. Our results show that MILES outperforms all baselines across all tasks and fusion methods considered in our study, effectively balancing modality usage during training. This results in improved multimodal performance and stronger modality encoders, which can be leveraged when dealing with unimodal samples or absent modalities. Overall, our work highlights the impact of balancing multimodal learning on improving model performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, training of multimodal networks is typically hindered by modality overfitting, where the network relies excessively on one of the available modalities.
* **Signal Tags:** #ai

---


### Rough Path Signatures: Learning Neural RDEs for Portfolio Optimization
**Date:** 2025-10-15 | **Arxiv:** [2510.10728](https://arxiv.org/abs/2510.10728)

#### Abstract
We tackle high-dimensional, path-dependent valuation and control and introduce a deep BSDE/2BSDE solver that couples truncated log-signatures with a neural rough differential equation (RDE) backbone. The architecture aligns stochastic analysis with sequence-to-path learning: a CVaR-tilted terminal objective targets left-tail risk, while an optional second-order (2BSDE) head supplies curvature estimates for risk-sensitive control. Under matched compute and parameter budgets, the method improves accuracy, tail fidelity, and training stability across Asian and barrier option pricing and portfolio control: at d=200 it achieves CVaR(0.99)=9.80% versus 12.00-13.10% for strong baselines, attains the lowest HJB residual (0.011), and yields the lowest RMSEs for Z and Gamma. Ablations over truncation depth, local windows, and tilt parameters confirm complementary gains from the sequence-to-path representation and the 2BSDE head. Taken together, the results highlight a bidirectional dialogue between stochastic analysis and modern deep learning: stochastic tools inform representations and objectives, while sequence-to-path models expand the class of solvable financial models at scale.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### In-Context Learning Is Provably Bayesian Inference: A Generalization Theory for Meta-Learning
**Date:** 2025-10-14 | **Arxiv:** [2510.10981](https://arxiv.org/abs/2510.10981)

#### Abstract
This paper develops a finite-sample statistical theory for in-context learning (ICL), analyzed within a meta-learning framework that accommodates mixtures of diverse task types. We introduce a principled risk decomposition that separates the total ICL risk into two orthogonal components: Bayes Gap and Posterior Variance. The Bayes Gap quantifies how well the trained model approximates the Bayes-optimal in-context predictor. For a uniform-attention Transformer, we derive a non-asymptotic upper bound on this gap, which explicitly clarifies the dependence on the number of pretraining prompts and their context length. The Posterior Variance is a model-independent risk representing the intrinsic task uncertainty. Our key finding is that this term is determined solely by the difficulty of the true underlying task, while the uncertainty arising from the task mixture vanishes exponentially fast with only a few in-context examples. Together, these results provide a unified view of ICL: the Transformer selects the optimal meta-algorithm during pretraining and rapidly converges to the optimal algorithm for the true task at test time.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks
**Date:** 2025-10-14 | **Arxiv:** [2510.11354](https://arxiv.org/abs/2510.11354)

#### Abstract
Adam is a popular and widely used adaptive gradient method in deep learning, which has also received tremendous focus in theoretical research. However, most existing theoretical work primarily analyzes its full-batch version, which differs fundamentally from the stochastic variant used in practice. Unlike SGD, stochastic Adam does not converge to its full-batch counterpart even with infinitesimal learning rates. We present the first theoretical characterization of how batch size affects Adam's generalization, analyzing two-layer over-parameterized CNNs on image data. Our results reveal that while both Adam and AdamW with proper weight decay $λ$ converge to poor test error solutions, their mini-batch variants can achieve near-zero test error. We further prove Adam has a strictly smaller effective weight decay bound than AdamW, theoretically explaining why Adam requires more sensitive $λ$ tuning. Extensive experiments validate our findings, demonstrating the critical role of batch size and weight decay in Adam's generalization performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, most existing theoretical work primarily analyzes its full-batch version, which differs fundamentally from the stochastic variant used in practice.
* **Signal Tags:** #ai

---


### Accuracy, Memory Efficiency and Generalization: A Comparative Study on Liquid Neural Networks and Recurrent Neural Networks
**Date:** 2025-10-10 | **Arxiv:** [2510.07578](https://arxiv.org/abs/2510.07578)

#### Abstract
This review aims to conduct a comparative analysis of liquid neural networks (LNNs) and traditional recurrent neural networks (RNNs) and their variants, such as long short-term memory networks (LSTMs) and gated recurrent units (GRUs). The core dimensions of the analysis include model accuracy, memory efficiency, and generalization ability. By systematically reviewing existing research, this paper explores the basic principles, mathematical models, key characteristics, and inherent challenges of these neural network architectures in processing sequential data. Research findings reveal that LNN, as an emerging, biologically inspired, continuous-time dynamic neural network, demonstrates significant potential in handling noisy, non-stationary data, and achieving out-of-distribution (OOD) generalization. Additionally, some LNN variants outperform traditional RNN in terms of parameter efficiency and computational speed. However, RNN remains a cornerstone in sequence modeling due to its mature ecosystem and successful applications across various tasks. This review identifies the commonalities and differences between LNNs and RNNs, summarizes their respective shortcomings and challenges, and points out valuable directions for future research, particularly emphasizing the importance of improving the scalability of LNNs to promote their application in broader and more complex scenarios.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, RNN remains a cornerstone in sequence modeling due to its mature ecosystem and successful applications across various tasks.
* **Signal Tags:** #ai

---


### Dynamic Learning Rate for Deep Reinforcement Learning: A Bandit Approach
**Date:** 2025-10-09 | **Arxiv:** [2410.12598](https://arxiv.org/abs/2410.12598)

#### Abstract
In deep Reinforcement Learning (RL), the learning rate critically influences both stability and performance, yet its optimal value shifts during training as the environment and policy evolve. Standard decay schedulers assume monotonic convergence and often misalign with these dynamics, leading to premature or delayed adjustments. We introduce LRRL, a meta-learning approach that dynamically selects the learning rate based on policy performance rather than training steps. LRRL adaptively favors rates that improve returns, remaining robust even when the candidate set includes values that individually cause divergence. Across Atari and MuJoCo benchmarks, LRRL achieves performance competitive with or superior to tuned baselines and standard schedulers. Our findings position LRRL as a practical solution for adapting to non-stationary objectives in deep RL.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization
**Date:** 2025-10-07 | **Arxiv:** [2402.14048](https://arxiv.org/abs/2402.14048)

#### Abstract
Reinforcement learning-based methods for constructing solutions to combinatorial optimization problems are rapidly approaching the performance of human-designed algorithms. To further narrow the gap, learning-based approaches must efficiently explore the solution space during the search process. Recent approaches artificially increase exploration by enforcing diverse solution generation through handcrafted rules, however, these rules can impair solution quality and are difficult to design for more complex problems. In this paper, we introduce PolyNet, an approach for improving exploration of the solution space by learning complementary solution strategies. In contrast to other works, PolyNet uses only a single-decoder and a training schema that does not enforce diverse solution generation through handcrafted rules. We evaluate PolyNet on four combinatorial optimization problems and observe that the implicit diversity mechanism allows PolyNet to find better solutions than approaches that explicitly enforce diverse solution generation.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** however, these rules can impair solution quality and are difficult to design for more complex problems.
* **Signal Tags:** #ai

---


### Q-Learning with Shift-Aware Upper Confidence Bound in Non-Stationary Reinforcement Learning
**Date:** 2025-10-06 | **Arxiv:** [2510.03181](https://arxiv.org/abs/2510.03181)

#### Abstract
We study the Non-Stationary Reinforcement Learning (RL) under distribution shifts in both finite-horizon episodic and infinite-horizon discounted Markov Decision Processes (MDPs). In the finite-horizon case, the transition functions may suddenly change at a particular episode. In the infinite-horizon setting, such changes can occur at an arbitrary time step during the agent's interaction with the environment. While the Q-learning Upper Confidence Bound algorithm (QUCB) can discover a proper policy during learning, due to the distribution shifts, this policy can exploit sub-optimal rewards after the shift happens. To address this issue, we propose Density-QUCB (DQUCB), a shift-aware Q-learning~UCB algorithm, which uses a transition density function to detect distribution shifts, then leverages its likelihood to enhance the uncertainty estimation quality of Q-learning~UCB, resulting in a balance between exploration and exploitation. Theoretically, we prove that our oracle DQUCB achieves a better regret guarantee than QUCB. Empirically, our DQUCB enjoys the computational efficiency of model-free RL and outperforms QUCB baselines by having a lower regret across RL tasks, as well as a real-world COVID-19 patient hospital allocation task using a Deep-Q-learning architecture.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### On the Emergence of Weak-to-Strong Generalization: A Bias-Variance Perspective
**Date:** 2025-09-30 | **Arxiv:** [2505.24313](https://arxiv.org/abs/2505.24313)

#### Abstract
Weak-to-strong generalization (W2SG) refers to the phenomenon where a strong student model, trained on a dataset labeled by a weak teacher, ultimately outperforms the teacher on the target task. Recent studies attribute this performance gain to the prediction misfit between the student and teacher models. In this work, we theoretically investigate the emergence of W2SG through a generalized bias-variance decomposition of Bregman divergence. Specifically, we show that the expected population risk gap between the student and teacher is quantified by the expected misfit between the two models. While this aligns with previous results, our analysis removes several restrictive assumptions, most notably, the convexity of the student's hypothesis class, required in earlier works. Moreover, we show that W2SG is more likely to emerge when the student model approximates its posterior mean teacher, rather than mimicking an individual teacher. Using a concrete example, we demonstrate that if the student model size is sufficiently large, it can indeed converge to the posterior mean teacher in expectation. Our analysis also suggests that avoiding overfitting to the teacher's supervision and reducing the entropy of student's prediction further facilitate W2SG. In addition, we show that the reverse cross-entropy loss, unlike the standard forward cross-entropy, is less sensitive to the predictive uncertainty of the teacher. Finally, we empirically verify our theoretical insights and demonstrate that incorporating the reverse cross-entropy loss consistently improves student performance.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### From Formal Language Theory to Statistical Learning: Finite Observability of Subregular Languages
**Date:** 2025-09-29 | **Arxiv:** [2509.22598](https://arxiv.org/abs/2509.22598)

#### Abstract
We prove that all standard subregular language classes are linearly separable when represented by their deciding predicates. This establishes finite observability and guarantees learnability with simple linear models. Synthetic experiments confirm perfect separability under noise-free conditions, while real-data experiments on English morphology show that learned features align with well-known linguistic constraints. These results demonstrate that the subregular hierarchy provides a rigorous and interpretable foundation for modeling natural language structure. Our code used in real-data experiments is available at https://github.com/UTokyo-HayashiLab/subregular.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### From Learning to Optimize to Learning Optimization Algorithms
**Date:** 2025-09-19 | **Arxiv:** [2405.18222](https://arxiv.org/abs/2405.18222)

#### Abstract
Towards designing learned optimization algorithms that are usable beyond their training setting, we identify key principles that classical algorithms obey, but have up to now, not been used for Learning to Optimize (L2O). Following these principles, we provide a general design pipeline, taking into account data, architecture and learning strategy, and thereby enabling a synergy between classical optimization and L2O, resulting in a philosophy of Learning Optimization Algorithms. As a consequence our learned algorithms perform well far beyond problems from the training distribution. We demonstrate the success of these novel principles by designing a new learning-enhanced BFGS algorithm and provide numerical experiments evidencing its adaptation to many settings at test time.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
