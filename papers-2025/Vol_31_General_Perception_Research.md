# Vol 31 General Perception Research
*Enriched by BITCOREOS | Phase 4 Batch 7*

---

### The Geometry of Intelligence: Deterministic Functional Topology as a Foundation for Real-World Perception
**Date:** 2025-12-05 | **Arxiv:** [2512.05089](https://arxiv.org/abs/2512.05089)

#### Abstract
Real-world physical processes do not generate arbitrary variability: their signals concentrate on compact and low-variability subsets of functional space. This geometric structure enables rapid generalization from a few examples in both biological and artificial systems.   This work develops a deterministic functional-topological framework in which the set of valid realizations of a physical phenomenon forms a compact perceptual manifold with stable invariants and a finite Hausdorff radius. We show that the boundaries of this manifold can be discovered in a fully self-supervised manner through Monte Carlo sampling, even when the governing equations of the system are unknown.   We provide theoretical guarantees, practical estimators of knowledge boundaries, and empirical validations across three domains: electromechanical railway point machines, electrochemical battery discharge curves, and physiological ECG signals.   Our results demonstrate that deterministic functional topology offers a unified mathematical foundation for perception, representation, and world-model construction, explaining why biological learners and self-supervised AI models can generalize from limited observations.

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


### ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness
**Date:** 2025-11-11 | **Arxiv:** [2504.10514](https://arxiv.org/abs/2504.10514)

#### Abstract
Color plays an important role in human perception and usually provides critical clues in visual reasoning. However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans. This paper introduces ColorBench, an innovative benchmark meticulously crafted to assess the capabilities of VLMs in color understanding, including color perception, reasoning, and robustness. By curating a suite of diverse test scenarios, with grounding in real applications, ColorBench evaluates how these models perceive colors, infer meanings from color-based cues, and maintain consistent performance under varying color transformations. Through an extensive evaluation of 32 VLMs with varying language models and vision encoders, our paper reveals some undiscovered findings: (i) The scaling law (larger models are better) still holds on ColorBench, while the language model plays a more important role than the vision encoder. (ii) However, the performance gaps across models are relatively small, indicating that color understanding has been largely neglected by existing VLMs. (iii) CoT reasoning improves color understanding accuracies and robustness, though they are vision-centric tasks. (iv) Color clues are indeed leveraged by VLMs on ColorBench but they can also mislead models in some tasks. These findings highlight the critical limitations of current VLMs and underscore the need to enhance color comprehension. Our ColorBenchcan serve as a foundational tool for advancing the study of human-level color understanding of multimodal AI.

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
* **Limits:** However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans.
* **Signal Tags:** #ai

---


### M2H: Multi-Task Learning with Efficient Window-Based Cross-Task Attention for Monocular Spatial Perception
**Date:** 2025-10-21 | **Arxiv:** [2510.17363](https://arxiv.org/abs/2510.17363)

#### Abstract
Deploying real-time spatial perception on edge devices requires efficient multi-task models that leverage complementary task information while minimizing computational overhead. This paper introduces Multi-Mono-Hydra (M2H), a novel multi-task learning framework designed for semantic segmentation and depth, edge, and surface normal estimation from a single monocular image. Unlike conventional approaches that rely on independent single-task models or shared encoder-decoder architectures, M2H introduces a Window-Based Cross-Task Attention Module that enables structured feature exchange while preserving task-specific details, improving prediction consistency across tasks. Built on a lightweight ViT-based DINOv2 backbone, M2H is optimized for real-time deployment and serves as the foundation for monocular spatial perception systems supporting 3D scene graph construction in dynamic environments. Comprehensive evaluations show that M2H outperforms state-of-the-art multi-task models on NYUDv2, surpasses single-task depth and semantic baselines on Hypersim, and achieves superior performance on the Cityscapes dataset, all while maintaining computational efficiency on laptop hardware. Beyond benchmarks, M2H is validated on real-world data, demonstrating its practicality in spatial perception tasks.

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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### Learning to Navigate Socially Through Proactive Risk Perception
**Date:** 2025-10-10 | **Arxiv:** [2510.07871](https://arxiv.org/abs/2510.07871)

#### Abstract
In this report, we describe the technical details of our submission to the IROS 2025 RoboSense Challenge Social Navigation Track. This track focuses on developing RGBD-based perception and navigation systems that enable autonomous agents to navigate safely, efficiently, and socially compliantly in dynamic human-populated indoor environments. The challenge requires agents to operate from an egocentric perspective using only onboard sensors including RGB-D observations and odometry, without access to global maps or privileged information, while maintaining social norm compliance such as safe distances and collision avoidance. Building upon the Falcon model, we introduce a Proactive Risk Perception Module to enhance social navigation performance. Our approach augments Falcon with collision risk understanding that learns to predict distance-based collision risk scores for surrounding humans, which enables the agent to develop more robust spatial awareness and proactive collision avoidance behaviors. The evaluation on the Social-HM3D benchmark demonstrates that our method improves the agent's ability to maintain personal space compliance while navigating toward goals in crowded indoor scenes with dynamic human agents, achieving 2nd place among 16 participating teams in the challenge.

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


### Generation is Required for Data-Efficient Perception
**Date:** 2025-12-10 | **Arxiv:** [2512.08854](https://arxiv.org/abs/2512.08854)

#### Abstract
It has been hypothesized that human-level visual perception requires a generative approach in which internal representations result from inverting a decoder. Yet today's most successful vision models are non-generative, relying on an encoder that maps images to representations without decoder inversion. This raises the question of whether generation is, in fact, necessary for machines to achieve human-level visual perception. To address this, we study whether generative and non-generative methods can achieve compositional generalization, a hallmark of human perception. Under a compositional data generating process, we formalize the inductive biases required to guarantee compositional generalization in decoder-based (generative) and encoder-based (non-generative) methods. We then show theoretically that enforcing these inductive biases on encoders is generally infeasible using regularization or architectural constraints. In contrast, for generative methods, the inductive biases can be enforced straightforwardly, thereby enabling compositional generalization by constraining a decoder and inverting it. We highlight how this inversion can be performed efficiently, either online through gradient-based search or offline through generative replay. We examine the empirical implications of our theory by training a range of generative and non-generative methods on photorealistic image datasets. We find that, without the necessary inductive biases, non-generative methods often fail to generalize compositionally and require large-scale pretraining or added supervision to improve generalization. By comparison, generative methods yield significant improvements in compositional generalization, without requiring additional data, by leveraging suitable inductive biases on a decoder along with search and replay.

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


### TranSimHub:A Unified Air-Ground Simulation Platform for Multi-Modal Perception and Decision-Making
**Date:** 2025-10-20 | **Arxiv:** [2510.15365](https://arxiv.org/abs/2510.15365)

#### Abstract
Air-ground collaborative intelligence is becoming a key approach for next-generation urban intelligent transportation management, where aerial and ground systems work together on perception, communication, and decision-making. However, the lack of a unified multi-modal simulation environment has limited progress in studying cross-domain perception, coordination under communication constraints, and joint decision optimization. To address this gap, we present TranSimHub, a unified simulation platform for air-ground collaborative intelligence. TranSimHub offers synchronized multi-view rendering across RGB, depth, and semantic segmentation modalities, ensuring consistent perception between aerial and ground viewpoints. It also supports information exchange between the two domains and includes a causal scene editor that enables controllable scenario creation and counterfactual analysis under diverse conditions such as different weather, emergency events, and dynamic obstacles. We release TranSimHub as an open-source platform that supports end-to-end research on perception, fusion, and control across realistic air and ground traffic scenes. Our code is available at https://github.com/Traffic-Alpha/TransSimHub.

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
* **Limits:** However, the lack of a unified multi-modal simulation environment has limited progress in studying cross-domain perception, coordination under communication constraints, and joint decision optimization.
* **Signal Tags:** #ai

---


### Uncolorable Examples: Preventing Unauthorized AI Colorization via Perception-Aware Chroma-Restrictive Perturbation
**Date:** 2025-10-13 | **Arxiv:** [2510.08979](https://arxiv.org/abs/2510.08979)

#### Abstract
AI-based colorization has shown remarkable capability in generating realistic color images from grayscale inputs. However, it poses risks of copyright infringement -- for example, the unauthorized colorization and resale of monochrome manga and films. Despite these concerns, no effective method currently exists to prevent such misuse. To address this, we introduce the first defensive paradigm, Uncolorable Examples, which embed imperceptible perturbations into grayscale images to invalidate unauthorized colorization. To ensure real-world applicability, we establish four criteria: effectiveness, imperceptibility, transferability, and robustness. Our method, Perception-Aware Chroma-Restrictive Perturbation (PAChroma), generates Uncolorable Examples that meet these four criteria by optimizing imperceptible perturbations with a Laplacian filter to preserve perceptual quality, and applying diverse input transformations during optimization to enhance transferability across models and robustness against common post-processing (e.g., compression). Experiments on ImageNet and Danbooru datasets demonstrate that PAChroma effectively degrades colorization quality while maintaining the visual appearance. This work marks the first step toward protecting visual content from illegitimate AI colorization, paving the way for copyright-aware defenses in generative media.

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
* **Limits:** However, it poses risks of copyright infringement -- for example, the unauthorized colorization and resale of monochrome manga and films.
* **Signal Tags:** #ai

---


### Inconsistent Affective Reaction: Sentiment of Perception and Opinion in Urban Environments
**Date:** 2025-10-10 | **Arxiv:** [2510.07359](https://arxiv.org/abs/2510.07359)

#### Abstract
The ascension of social media platforms has transformed our understanding of urban environments, giving rise to nuanced variations in sentiment reaction embedded within human perception and opinion, and challenging existing multidimensional sentiment analysis approaches in urban studies. This study presents novel methodologies for identifying and elucidating sentiment inconsistency, constructing a dataset encompassing 140,750 Baidu and Tencent Street view images to measure perceptions, and 984,024 Weibo social media text posts to measure opinions. A reaction index is developed, integrating object detection and natural language processing techniques to classify sentiment in Beijing Second Ring for 2016 and 2022. Classified sentiment reaction is analysed and visualized using regression analysis, image segmentation, and word frequency based on land-use distribution to discern underlying factors. The perception affective reaction trend map reveals a shift toward more evenly distributed positive sentiment, while the opinion affective reaction trend map shows more extreme changes. Our mismatch map indicates significant disparities between the sentiments of human perception and opinion of urban areas over the years. Changes in sentiment reactions have significant relationships with elements such as dense buildings and pedestrian presence. Our inconsistent maps present perception and opinion sentiments before and after the pandemic and offer potential explanations and directions for environmental management, in formulating strategies for urban renewal.

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


### User eXperience Perception Insights Dataset (UXPID): Synthetic User Feedback from Public Industrial Forums
**Date:** 2025-09-16 | **Arxiv:** [2509.11777](https://arxiv.org/abs/2509.11777)

#### Abstract
Customer feedback in industrial forums reflect a rich but underexplored source of insight into real-world product experience. These publicly shared discussions offer an organic view of user expectations, frustrations, and success stories shaped by the specific contexts of use. Yet, harnessing this information for systematic analysis remains challenging due to the unstructured and domain-specific nature of the content. The lack of structure and specialized vocabulary makes it difficult for traditional data analysis techniques to accurately interpret, categorize, and quantify the feedback, thereby limiting its potential to inform product development and support strategies. To address these challenges, this paper presents the User eXperience Perception Insights Dataset (UXPID), a collection of 7130 artificially synthesized and anonymized user feedback branches extracted from a public industrial automation forum. Each JavaScript object notation (JSON) record contains multi-post comments related to specific hardware and software products, enriched with metadata and contextual conversation data. Leveraging a large language model (LLM), each branch is systematically analyzed and annotated for UX insights, user expectations, severity and sentiment ratings, and topic classifications. The UXPID dataset is designed to facilitate research in user requirements, user experience (UX) analysis, and AI-driven feedback processing, particularly where privacy and licensing restrictions limit access to real-world data. UXPID supports the training and evaluation of transformer-based models for tasks such as issue detection, sentiment analysis, and requirements extraction in the context of technical forums.

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
* **Layer:** Hardware
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### The GOOSE Dataset for Perception in Unstructured Environments
**Date:** 2025-09-09 | **Arxiv:** [2310.16788](https://arxiv.org/abs/2310.16788)

#### Abstract
The potential for deploying autonomous systems can be significantly increased by improving the perception and interpretation of the environment. However, the development of deep learning-based techniques for autonomous systems in unstructured outdoor environments poses challenges due to limited data availability for training and testing. To address this gap, we present the German Outdoor and Offroad Dataset (GOOSE), a comprehensive dataset specifically designed for unstructured outdoor environments. The GOOSE dataset incorporates 10 000 labeled pairs of images and point clouds, which are utilized to train a range of state-of-the-art segmentation models on both image and point cloud data. We open source the dataset, along with an ontology for unstructured terrain, as well as dataset standards and guidelines. This initiative aims to establish a common framework, enabling the seamless inclusion of existing datasets and a fast way to enhance the perception capabilities of various robots operating in unstructured environments. The dataset, pre-trained models for offroad perception, and additional documentation can be found at https://goose-dataset.de/.

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
* **Limits:** However, the development of deep learning-based techniques for autonomous systems in unstructured outdoor environments poses challenges due to limited data availability for training and testing.
* **Signal Tags:** #ai

---


### CoVeRaP: Cooperative Vehicular Perception through mmWave FMCW Radars
**Date:** 2025-08-25 | **Arxiv:** [2508.16030](https://arxiv.org/abs/2508.16030)

#### Abstract
Automotive FMCW radars remain reliable in rain and glare, yet their sparse, noisy point clouds constrain 3-D object detection. We therefore release CoVeRaP, a 21 k-frame cooperative dataset that time-aligns radar, camera, and GPS streams from multiple vehicles across diverse manoeuvres. Built on this data, we propose a unified cooperative-perception framework with middle- and late-fusion options. Its baseline network employs a multi-branch PointNet-style encoder enhanced with self-attention to fuse spatial, Doppler, and intensity cues into a common latent space, which a decoder converts into 3-D bounding boxes and per-point depth confidence. Experiments show that middle fusion with intensity encoding boosts mean Average Precision by up to 9x at IoU 0.9 and consistently outperforms single-vehicle baselines. CoVeRaP thus establishes the first reproducible benchmark for multi-vehicle FMCW-radar perception and demonstrates that affordable radar sharing markedly improves detection robustness. Dataset and code are publicly available to encourage further research.

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
