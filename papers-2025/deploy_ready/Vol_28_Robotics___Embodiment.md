# Vol 28 Robotics   Embodiment
*Enriched by BITCOREOS | Phase 4 Batch 6*

---

### Semi Centralized Training Decentralized Execution Architecture for Multi Agent Deep Reinforcement Learning in Traffic Signal Control
**Date:** 2025-12-05 | **Arxiv:** [2512.04653](https://hub.bitwiki.org/t/semi-centralized-training-decentralized-execution-architecture-for-multi-agent-deep-reinforcement-learning-in-traffic-signal-control/27702)

#### Abstract
Multi-agent reinforcement learning (MARL) has emerged as a promising paradigm for adaptive traffic signal control (ATSC) of multiple intersections. Existing approaches typically follow either a fully centralized or a fully decentralized design. Fully centralized approaches suffer from the curse of dimensionality, and reliance on a single learning server, whereas purely decentralized approaches operate under severe partial observability and lack explicit coordination resulting in suboptimal performance. These limitations motivate region-based MARL, where the network is partitioned into smaller, tightly coupled intersections that form regions, and training is organized around these regions. This paper introduces a Semi-Centralized Training, Decentralized Execution (SEMI-CTDE) architecture for multi intersection ATSC. Within each region, SEMI-CTDE performs centralized training with regional parameter sharing and employs composite state and reward formulations that jointly encode local and regional information. The architecture is highly transferable across different policy backbones and state-reward instantiations. Building on this architecture, we implement two models with distinct design objectives. A multi-perspective experimental analysis of the two implemented SEMI-CTDE-based models covering ablations of the architecture's core elements including rule based and fully decentralized baselines shows that they achieve consistently superior performance and remain effective across a wide range of traffic densities and distributions.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Learnable Conformal Prediction with Context-Aware Nonconformity Functions for Robotic Planning and Perception
**Date:** 2025-09-29 | **Arxiv:** [2509.21955](https://hub.bitwiki.org/t/learnable-conformal-prediction-with-context-aware-nonconformity-functions-for-robotic-planning-and-perception/12190)

#### Abstract
Deep learning models in robotics often output point estimates with poorly calibrated confidences, offering no native mechanism to quantify predictive reliability under novel, noisy, or out-of-distribution inputs. Conformal prediction (CP) addresses this gap by providing distribution-free coverage guarantees, yet its reliance on fixed nonconformity scores ignores context and can yield intervals that are overly conservative or unsafe. We address this with Learnable Conformal Prediction (LCP), which replaces fixed scores with a lightweight neural function that leverages geometric, semantic, and task-specific features to produce context-aware uncertainty sets.   LCP maintains CP's theoretical guarantees while reducing prediction set sizes by 18% in classification, tightening detection intervals by 52%, and improving path planning safety from 72% to 91% success with minimal overhead. Across three robotic tasks on seven benchmarks, LCP consistently outperforms Standard CP and ensemble baselines. In classification on CIFAR-100 and ImageNet, it achieves smaller set sizes (4.7-9.9% reduction) at target coverage. For object detection on COCO, BDD100K, and Cityscapes, it produces 46-54% tighter bounding boxes. In path planning through cluttered environments, it improves success to 91.5% with only 4.5% path inflation, compared to 12.2% for Standard CP.   The method is lightweight (approximately 4.8% runtime overhead, 42 KB memory) and supports online adaptation, making it well suited to resource-constrained autonomous systems. Hardware evaluation shows LCP adds less than 1% memory and 15.9% inference overhead, yet sustains 39 FPS on detection tasks while being 7.4 times more energy-efficient than ensembles.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Investigating Robot Control Policy Learning for Autonomous X-ray-guided Spine Procedures
**Date:** 2025-11-07 | **Arxiv:** [2511.03882](https://hub.bitwiki.org/t/investigating-robot-control-policy-learning-for-autonomous-x-ray-guided-spine-procedures/22096)

#### Abstract
Imitation learning-based robot control policies are enjoying renewed interest in video-based robotics. However, it remains unclear whether this approach applies to X-ray-guided procedures, such as spine instrumentation. This is because interpretation of multi-view X-rays is complex. We examine opportunities and challenges for imitation policy learning in bi-plane-guided cannula insertion. We develop an in silico sandbox for scalable, automated simulation of X-ray-guided spine procedures with a high degree of realism. We curate a dataset of correct trajectories and corresponding bi-planar X-ray sequences that emulate the stepwise alignment of providers. We then train imitation learning policies for planning and open-loop control that iteratively align a cannula solely based on visual information. This precisely controlled setup offers insights into limitations and capabilities of this method. Our policy succeeded on the first attempt in 68.5% of cases, maintaining safe intra-pedicular trajectories across diverse vertebral levels. The policy generalized to complex anatomy, including fractures, and remained robust to varied initializations. Rollouts on real bi-planar X-rays further suggest that the model can produce plausible trajectories, despite training exclusively in simulation. While these preliminary results are promising, we also identify limitations, especially in entry point precision. Full closed-look control will require additional considerations around how to provide sufficiently frequent feedback. With more robust priors and domain knowledge, such models may provide a foundation for future efforts toward lightweight and CT-free robotic intra-operative spinal navigation.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, it remains unclear whether this approach applies to X-ray-guided procedures, such as spine instrumentation.
* **Signal Tags:** #ai

---


### Real-Time Learning of Predictive Dynamic Obstacle Models for Robotic Motion Planning
**Date:** 2025-11-04 | **Arxiv:** [2511.00814](https://hub.bitwiki.org/t/real-time-learning-of-predictive-dynamic-obstacle-models-for-robotic-motion-planning/21307)

#### Abstract
Autonomous systems often must predict the motions of nearby agents from partial and noisy data. This paper asks and answers the question: "can we learn, in real-time, a nonlinear predictive model of another agent's motions?" Our online framework denoises and forecasts such dynamics using a modified sliding-window Hankel Dynamic Mode Decomposition (Hankel-DMD). Partial noisy measurements are embedded into a Hankel matrix, while an associated Page matrix enables singular-value hard thresholding (SVHT) to estimate the effective rank. A Cadzow projection enforces structured low-rank consistency, yielding a denoised trajectory and local noise variance estimates. From this representation, a time-varying Hankel-DMD lifted linear predictor is constructed for multi-step forecasts. The residual analysis provides variance-tracking signals that can support downstream estimators and risk-aware planning. We validate the approach in simulation under Gaussian and heavy-tailed noise, and experimentally on a dynamic crane testbed. Results show that the method achieves stable variance-aware denoising and short-horizon prediction suitable for integration into real-time control frameworks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### ROPES: Robotic Pose Estimation via Score-Based Causal Representation Learning
**Date:** 2025-10-27 | **Arxiv:** [2510.20884](https://hub.bitwiki.org/t/ropes-robotic-pose-estimation-via-score-based-causal-representation-learning/19584)

#### Abstract
Causal representation learning (CRL) has emerged as a powerful unsupervised framework that (i) disentangles the latent generative factors underlying high-dimensional data, and (ii) learns the cause-and-effect interactions among the disentangled variables. Despite extensive recent advances in identifiability and some practical progress, a substantial gap remains between theory and real-world practice. This paper takes a step toward closing that gap by bringing CRL to robotics, a domain that has motivated CRL. Specifically, this paper addresses the well-defined robot pose estimation -- the recovery of position and orientation from raw images -- by introducing Robotic Pose Estimation via Score-Based CRL (ROPES). Being an unsupervised framework, ROPES embodies the essence of interventional CRL by identifying those generative factors that are actuated: images are generated by intrinsic and extrinsic latent factors (e.g., joint angles, arm/limb geometry, lighting, background, and camera configuration) and the objective is to disentangle and recover the controllable latent variables, i.e., those that can be directly manipulated (intervened upon) through actuation. Interventional CRL theory shows that variables that undergo variations via interventions can be identified. In robotics, such interventions arise naturally by commanding actuators of various joints and recording images under varied controls. Empirical evaluations in semi-synthetic manipulator experiments demonstrate that ROPES successfully disentangles latent generative factors with high fidelity with respect to the ground truth. Crucially, this is achieved by leveraging only distributional changes, without using any labeled data. The paper also includes a comparison with a baseline based on a recently proposed semi-supervised framework. This paper concludes by positioning robot pose estimation as a near-practical testbed for CRL.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Robust Driving Control for Autonomous Vehicles: An Intelligent General-sum Constrained Adversarial Reinforcement Learning Approach
**Date:** 2025-10-13 | **Arxiv:** [2510.09041](https://hub.bitwiki.org/t/robust-driving-control-for-autonomous-vehicles-an-intelligent-general-sum-constrained-adversarial-reinforcement-learning-approach/16304)

#### Abstract
Deep reinforcement learning (DRL) has demonstrated remarkable success in developing autonomous driving policies. However, its vulnerability to adversarial attacks remains a critical barrier to real-world deployment. Although existing robust methods have achieved success, they still suffer from three key issues: (i) these methods are trained against myopic adversarial attacks, limiting their abilities to respond to more strategic threats, (ii) they have trouble causing truly safety-critical events (e.g., collisions), but instead often result in minor consequences, and (iii) these methods can introduce learning instability and policy drift during training due to the lack of robust constraints. To address these issues, we propose Intelligent General-sum Constrained Adversarial Reinforcement Learning (IGCARL), a novel robust autonomous driving approach that consists of a strategic targeted adversary and a robust driving agent. The strategic targeted adversary is designed to leverage the temporal decision-making capabilities of DRL to execute strategically coordinated multi-step attacks. In addition, it explicitly focuses on inducing safety-critical events by adopting a general-sum objective. The robust driving agent learns by interacting with the adversary to develop a robust autonomous driving policy against adversarial attacks. To ensure stable learning in adversarial environments and to mitigate policy drift caused by attacks, the agent is optimized under a constrained formulation. Extensive experiments show that IGCARL improves the success rate by at least 27.9% over state-of-the-art methods, demonstrating superior robustness to adversarial attacks and enhancing the safety and reliability of DRL-based autonomous driving.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Limits:** However, its vulnerability to adversarial attacks remains a critical barrier to real-world deployment.
* **Signal Tags:** #ai

---


### Gated Uncertainty-Aware Runtime Dual Invariants for Neural Signal-Controlled Robotics
**Date:** 2025-11-26 | **Arxiv:** [2511.20570](https://hub.bitwiki.org/t/gated-uncertainty-aware-runtime-dual-invariants-for-neural-signal-controlled-robotics/26000)

#### Abstract
Safety-critical assistive systems that directly decode user intent from neural signals require rigorous guarantees of reliability and trust. We present GUARDIAN (Gated Uncertainty-Aware Runtime Dual Invariants), a framework for real-time neuro-symbolic verification for neural signal-controlled robotics. GUARDIAN enforces both logical safety and physiological trust by coupling confidence-calibrated brain signal decoding with symbolic goal grounding and dual-layer runtime monitoring. On the BNCI2014 motor imagery electroencephalogram (EEG) dataset with 9 subjects and 5,184 trials, the system performs at a high safety rate of 94-97% even with lightweight decoder architectures with low test accuracies (27-46%) and high ECE confidence miscalibration (0.22-0.41). We demonstrate 1.7x correct interventions in simulated noise testing versus at baseline. The monitor operates at 100Hz and sub-millisecond decision latency, making it practically viable for closed-loop neural signal-based systems. Across 21 ablation results, GUARDIAN exhibits a graduated response to signal degradation, and produces auditable traces from intent, plan to action, helping to link neural evidence to verifiable robot action.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Adaptive Inverse Kinematics Framework for Learning Variable-Length Tool Manipulation in Robotics
**Date:** 2025-10-31 | **Arxiv:** [2510.26551](https://hub.bitwiki.org/t/adaptive-inverse-kinematics-framework-for-learning-variable-length-tool-manipulation-in-robotics/20655)

#### Abstract
Conventional robots possess a limited understanding of their kinematics and are confined to preprogrammed tasks, hindering their ability to leverage tools efficiently. Driven by the essential components of tool usage - grasping the desired outcome, selecting the most suitable tool, determining optimal tool orientation, and executing precise manipulations - we introduce a pioneering framework. Our novel approach expands the capabilities of the robot's inverse kinematics solver, empowering it to acquire a sequential repertoire of actions using tools of varying lengths. By integrating a simulation-learned action trajectory with the tool, we showcase the practicality of transferring acquired skills from simulation to real-world scenarios through comprehensive experimentation. Remarkably, our extended inverse kinematics solver demonstrates an impressive error rate of less than 1 cm. Furthermore, our trained policy achieves a mean error of 8 cm in simulation. Noteworthy, our model achieves virtually indistinguishable performance when employing two distinct tools of different lengths. This research provides an indication of potential advances in the exploration of all four fundamental aspects of tool usage, enabling robots to master the intricate art of tool manipulation across diverse tasks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Layer:** Infrastructure
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Debate2Create: Robot Co-design via Large Language Model Debates
**Date:** 2025-10-31 | **Arxiv:** [2510.25850](https://hub.bitwiki.org/t/debate2create-robot-co-design-via-large-language-model-debates/20623)

#### Abstract
Automating the co-design of a robot's morphology and control is a long-standing challenge due to the vast design space and the tight coupling between body and behavior. We introduce Debate2Create (D2C), a framework in which large language model (LLM) agents engage in a structured dialectical debate to jointly optimize a robot's design and its reward function. In each round, a design agent proposes targeted morphological modifications, and a control agent devises a reward function tailored to exploit the new design. A panel of pluralistic judges then evaluates the design-control pair in simulation and provides feedback that guides the next round of debate. Through iterative debates, the agents progressively refine their proposals, producing increasingly effective robot designs. Notably, D2C yields diverse and specialized morphologies despite no explicit diversity objective. On a quadruped locomotion benchmark, D2C discovers designs that travel 73% farther than the default, demonstrating that structured LLM-based debate can serve as a powerful mechanism for emergent robot co-design. Our results suggest that multi-agent debate, when coupled with physics-grounded feedback, is a promising new paradigm for automated robot design.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos
**Date:** 2025-10-17 | **Arxiv:** [2412.04445](https://hub.bitwiki.org/t/moto-latent-motion-token-as-the-bridging-language-for-learning-robot-manipulation-from-videos/17643)

#### Abstract
Recent developments in Large Language Models pre-trained on extensive corpora have shown significant success in various natural language processing tasks with minimal fine-tuning. This success offers new promise for robotics, which has long been constrained by the high cost of action-labeled data. We ask: given the abundant video data containing interaction-related knowledge available as a rich "corpus", can a similar generative pre-training approach be effectively applied to enhance robot learning? The key challenge is to identify an effective representation for autoregressive pre-training that benefits robot manipulation tasks. Inspired by the way humans learn new skills through observing dynamic environments, we propose that effective robotic learning should emphasize motion-related knowledge, which is closely tied to low-level actions and is hardware-agnostic, facilitating the transfer of learned motions to actual robot actions. To this end, we introduce Moto, which converts video content into latent Motion Token sequences by a Latent Motion Tokenizer, learning a bridging "language" of motion from videos in an unsupervised manner. We pre-train Moto-GPT through motion token autoregression, enabling it to capture diverse visual motion knowledge. After pre-training, Moto-GPT demonstrates the promising ability to produce semantically interpretable motion tokens, predict plausible motion trajectories, and assess trajectory rationality through output likelihood. To transfer learned motion priors to real robot actions, we implement a co-fine-tuning strategy that seamlessly bridges latent motion token prediction and real robot control. Extensive experiments show that the fine-tuned Moto-GPT exhibits superior robustness and efficiency on robot manipulation benchmarks, underscoring its effectiveness in transferring knowledge from video data to downstream visual manipulation tasks.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Medical Vision Language Models as Policies for Robotic Surgery
**Date:** 2025-10-08 | **Arxiv:** [2510.06064](https://hub.bitwiki.org/t/medical-vision-language-models-as-policies-for-robotic-surgery/15483)

#### Abstract
Vision-based Proximal Policy Optimization (PPO) struggles with visual observation-based robotic laparoscopic surgical tasks due to the high-dimensional nature of visual input, the sparsity of rewards in surgical environments, and the difficulty of extracting task-relevant features from raw visual data. We introduce a simple approach integrating MedFlamingo, a medical domain-specific Vision-Language Model, with PPO. Our method is evaluated on five diverse laparoscopic surgery task environments in LapGym, using only endoscopic visual observations. MedFlamingo PPO outperforms and converges faster compared to both standard vision-based PPO and OpenFlamingo PPO baselines, achieving task success rates exceeding 70% across all environments, with improvements ranging from 66.67% to 1114.29% compared to baseline. By processing task observations and instructions once per episode to generate high-level planning tokens, our method efficiently combines medical expertise with real-time visual feedback. Our results highlight the value of specialized medical knowledge in robotic surgical planning and decision-making.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Resolve Highway Conflict in Multi-Autonomous Vehicle Controls with Local State Attention
**Date:** 2025-09-19 | **Arxiv:** [2506.11445](https://hub.bitwiki.org/t/resolve-highway-conflict-in-multi-autonomous-vehicle-controls-with-local-state-attention/10108)

#### Abstract
In mixed-traffic environments, autonomous vehicles must adapt to human-controlled vehicles and other unusual driving situations. This setting can be framed as a multi-agent reinforcement learning (MARL) environment with full cooperative reward among the autonomous vehicles. While methods such as Multi-agent Proximal Policy Optimization can be effective in training MARL tasks, they often fail to resolve local conflict between agents and are unable to generalize to stochastic events. In this paper, we propose a Local State Attention module to assist the input state representation. By relying on the self-attention operator, the module is expected to compress the essential information of nearby agents to resolve the conflict in traffic situations. Utilizing a simulated highway merging scenario with the priority vehicle as the unexpected event, our approach is able to prioritize other vehicles' information to manage the merging process. The results demonstrate significant improvements in merging efficiency compared to popular baselines, especially in high-density traffic settings.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### NeRF-Aug: Data Augmentation for Robotics with Neural Radiance Fields
**Date:** 2025-09-16 | **Arxiv:** [2411.02482](https://hub.bitwiki.org/t/nerf-aug-data-augmentation-for-robotics-with-neural-radiance-fields/9707)

#### Abstract
Training a policy that can generalize to unknown objects is a long standing challenge within the field of robotics. The performance of a policy often drops significantly in situations where an object in the scene was not seen during training. To solve this problem, we present NeRF-Aug, a novel method that is capable of teaching a policy to interact with objects that are not present in the dataset. This approach differs from existing approaches by leveraging the speed, photorealism, and 3D consistency of a neural radiance field for augmentation. NeRF-Aug both creates more photorealistic data and runs 63% faster than existing methods. We demonstrate the effectiveness of our method on 5 tasks with 9 novel objects that are not present in the expert demonstrations. We achieve an average performance boost of 55.6% when comparing our method to the next best method. You can see video results at https://nerf-aug.github.io.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Solving Robotics Tasks with Prior Demonstration via Exploration-Efficient Deep Reinforcement Learning
**Date:** 2025-09-08 | **Arxiv:** [2509.04069](https://hub.bitwiki.org/t/solving-robotics-tasks-with-prior-demonstration-via-exploration-efficient-deep-reinforcement-learning/8100)

#### Abstract
This paper proposes an exploration-efficient Deep Reinforcement Learning with Reference policy (DRLR) framework for learning robotics tasks that incorporates demonstrations. The DRLR framework is developed based on an algorithm called Imitation Bootstrapped Reinforcement Learning (IBRL). We propose to improve IBRL by modifying the action selection module. The proposed action selection module provides a calibrated Q-value, which mitigates the bootstrapping error that otherwise leads to inefficient exploration. Furthermore, to prevent the RL policy from converging to a sub-optimal policy, SAC is used as the RL policy instead of TD3. The effectiveness of our method in mitigating bootstrapping error and preventing overfitting is empirically validated by learning two robotics tasks: bucket loading and open drawer, which require extensive interactions with the environment. Simulation results also demonstrate the robustness of the DRLR framework across tasks with both low and high state-action dimensions, and varying demonstration qualities. To evaluate the developed framework on a real-world industrial robotics task, the bucket loading task is deployed on a real wheel loader. The sim2real results validate the successful deployment of the DRLR framework.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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


### Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation
**Date:** 2025-08-20 | **Arxiv:** [2508.13998](https://hub.bitwiki.org/t/embodied-r1-reinforced-embodied-reasoning-for-general-robotic-manipulation/4812)

#### Abstract
Generalization in embodied AI is hindered by the "seeing-to-doing gap," which stems from data scarcity and embodiment heterogeneity. To address this, we pioneer "pointing" as a unified, embodiment-agnostic intermediate representation, defining four core embodied pointing abilities that bridge high-level vision-language comprehension with low-level action primitives. We introduce Embodied-R1, a 3B Vision-Language Model (VLM) specifically designed for embodied reasoning and pointing. We use a wide range of embodied and general visual reasoning datasets as sources to construct a large-scale dataset, Embodied-Points-200K, which supports key embodied pointing capabilities. We then train Embodied-R1 using a two-stage Reinforced Fine-tuning (RFT) curriculum with a specialized multi-task reward design. Embodied-R1 achieves state-of-the-art performance on 11 embodied spatial and pointing benchmarks. Critically, it demonstrates robust zero-shot generalization by achieving a 56.2% success rate in the SIMPLEREnv and 87.5% across 8 real-world XArm tasks without any task-specific fine-tuning, representing a 62% improvement over strong baselines. Furthermore, the model exhibits high robustness against diverse visual disturbances. Our work shows that a pointing-centric representation, combined with an RFT training paradigm, offers an effective and generalizable pathway to closing the perception-action gap in robotics.

#### Research Highlights
- **Core Innovation:** See abstract.
- **Methodology:** See abstract.
- **Key Finding:** See abstract.

#### Technical Context
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
