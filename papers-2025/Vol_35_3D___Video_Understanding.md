# Vol 35 3D   Video Understanding
*Enriched by BITCOREOS | Phase 4 Batch 7*

---

### VIDEOP2R: Video Understanding from Perception to Reasoning
**Date:** 2025-11-17 | **Arxiv:** [2511.11113](https://arxiv.org/abs/2511.11113)

#### Abstract
Reinforcement fine-tuning (RFT), a two-stage framework consisting of supervised fine-tuning (SFT) and reinforcement learning (RL) has shown promising results on improving reasoning ability of large language models (LLMs). Yet extending RFT to large video language models (LVLMs) remains challenging. We propose VideoP2R, a novel process-aware video RFT framework that enhances video reasoning by modeling perception and reasoning as distinct processes. In the SFT stage, we develop a three-step pipeline to generate VideoP2R-CoT-162K, a high-quality, process-aware chain-of-thought (CoT) dataset for perception and reasoning. In the RL stage, we introduce a novel process-aware group relative policy optimization (PA-GRPO) algorithm that supplies separate rewards for perception and reasoning. Extensive experiments show that VideoP2R achieves state-of-the-art (SotA) performance on six out of seven video reasoning and understanding benchmarks. Ablation studies further confirm the effectiveness of our process-aware modeling and PA-GRPO and demonstrate that model's perception output is information-sufficient for downstream reasoning.

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


### Rethinking Chain-of-Thought Reasoning for Videos
**Date:** 2025-12-11 | **Arxiv:** [2512.09616](https://arxiv.org/abs/2512.09616)

#### Abstract
Chain-of-thought (CoT) reasoning has been highly successful in solving complex tasks in natural language processing, and recent multimodal large language models (MLLMs) have extended this paradigm to video reasoning. However, these models typically build on lengthy reasoning chains and large numbers of input visual tokens. Motivated by empirical observations from our benchmark study, we hypothesize that concise reasoning combined with a reduced set of visual tokens can be sufficient for effective video reasoning. To evaluate this hypothesis, we design and validate an efficient post-training and inference framework that enhances a video MLLM's reasoning capability. Our framework enables models to operate on compressed visual tokens and generate brief reasoning traces prior to answering. The resulting models achieve substantially improved inference efficiency, deliver competitive performance across diverse benchmarks, and avoid reliance on manual CoT annotations or supervised fine-tuning. Collectively, our results suggest that long, human-like CoT reasoning may not be necessary for general video reasoning, and that concise reasoning can be both effective and efficient. Our code will be released at https://github.com/LaVi-Lab/Rethink_CoT_Video.

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
* **Limits:** However, these models typically build on lengthy reasoning chains and large numbers of input visual tokens.
* **Signal Tags:** #ai

---


### OmniX: From Unified Panoramic Generation and Perception to Graphics-Ready 3D Scenes
**Date:** 2025-10-31 | **Arxiv:** [2510.26800](https://arxiv.org/abs/2510.26800)

#### Abstract
There are two prevalent ways to constructing 3D scenes: procedural generation and 2D lifting. Among them, panorama-based 2D lifting has emerged as a promising technique, leveraging powerful 2D generative priors to produce immersive, realistic, and diverse 3D environments. In this work, we advance this technique to generate graphics-ready 3D scenes suitable for physically based rendering (PBR), relighting, and simulation. Our key insight is to repurpose 2D generative models for panoramic perception of geometry, textures, and PBR materials. Unlike existing 2D lifting approaches that emphasize appearance generation and ignore the perception of intrinsic properties, we present OmniX, a versatile and unified framework. Based on a lightweight and efficient cross-modal adapter structure, OmniX reuses 2D generative priors for a broad range of panoramic vision tasks, including panoramic perception, generation, and completion. Furthermore, we construct a large-scale synthetic panorama dataset containing high-quality multimodal panoramas from diverse indoor and outdoor scenes. Extensive experiments demonstrate the effectiveness of our model in panoramic visual perception and graphics-ready 3D scene generation, opening new possibilities for immersive and physically realistic virtual world generation.

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


### Video Reasoning without Training
**Date:** 2025-10-21 | **Arxiv:** [2510.17045](https://arxiv.org/abs/2510.17045)

#### Abstract
Video reasoning using Large Multimodal Models (LMMs) relies on costly reinforcement learning (RL) and verbose chain-of-thought, resulting in substantial computational overhead during both training and inference. Moreover, the mechanisms that control the thinking process in these reasoning models are very limited. In this paper, using entropy of the model's output as a signal, we discover that the high-quality models go through a series of micro-explorations and micro-exploitations which keep the reasoning process grounded (i.e., avoid excessive randomness while the model is exploring or thinking through an answer). We further observe that once this "thinking" process is over, more accurate models demonstrate a better convergence by reducing the entropy significantly via a final exploitation phase (i.e., a more certain convergence towards a solution trajectory). We then use these novel, theoretically-grounded insights to tune the model's behavior directly at inference, without using any RL or supervised fine-tuning. Specifically, during inference, our proposed approach called V-Reason (Video-Reason) adapts the value cache of the LMM via a few optimization steps on a small, trainable controller using an entropy-based objective, i.e., no supervision from any dataset or RL is necessary. This tuning improves the model's micro-exploration and exploitation behavior during inference. Our experiments show that our proposed method achieves significant improvements over the base instruction-tuned models across several video reasoning datasets, narrowing the gap with RL-trained models to within 0.6% average accuracy without any training, while offering massive efficiency benefits: output tokens are reduced by 58.6% compared to the RL model.

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


### Switchboard-Affect: Emotion Perception Labels from Conversational Speech
**Date:** 2025-10-17 | **Arxiv:** [2510.13906](https://arxiv.org/abs/2510.13906)

#### Abstract
Understanding the nuances of speech emotion dataset curation and labeling is essential for assessing speech emotion recognition (SER) model potential in real-world applications. Most training and evaluation datasets contain acted or pseudo-acted speech (e.g., podcast speech) in which emotion expressions may be exaggerated or otherwise intentionally modified. Furthermore, datasets labeled based on crowd perception often lack transparency regarding the guidelines given to annotators. These factors make it difficult to understand model performance and pinpoint necessary areas for improvement. To address this gap, we identified the Switchboard corpus as a promising source of naturalistic conversational speech, and we trained a crowd to label the dataset for categorical emotions (anger, contempt, disgust, fear, sadness, surprise, happiness, tenderness, calmness, and neutral) and dimensional attributes (activation, valence, and dominance). We refer to this label set as Switchboard-Affect (SWB-Affect). In this work, we present our approach in detail, including the definitions provided to annotators and an analysis of the lexical and paralinguistic cues that may have played a role in their perception. In addition, we evaluate state-of-the-art SER models, and we find variable performance across the emotion categories with especially poor generalization for anger. These findings underscore the importance of evaluation with datasets that capture natural affective variations in speech. We release the labels for SWB-Affect to enable further analysis in this domain.

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
* **Layer:** Application
* **Limits:** No explicit limitations stated.
* **Signal Tags:** #ai

---


### DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception
**Date:** 2025-09-15 | **Arxiv:** [2509.09828](https://arxiv.org/abs/2509.09828)

#### Abstract
Robust semantic perception for autonomous vehicles relies on effectively combining multiple sensors with complementary strengths and weaknesses. State-of-the-art sensor fusion approaches to semantic perception often treat sensor data uniformly across the spatial extent of the input, which hinders performance when faced with challenging conditions. By contrast, we propose a novel depth-guided multimodal fusion method that upgrades condition-aware fusion by integrating depth information. Our network, DGFusion, poses multimodal segmentation as a multi-task problem, utilizing the lidar measurements, which are typically available in outdoor sensor suites, both as one of the model's inputs and as ground truth for learning depth. Our corresponding auxiliary depth head helps to learn depth-aware features, which are encoded into spatially varying local depth tokens that condition our attentive cross-modal fusion. Together with a global condition token, these local depth tokens dynamically adapt sensor fusion to the spatially varying reliability of each sensor across the scene, which largely depends on depth. In addition, we propose a robust loss for our depth, which is essential for learning from lidar inputs that are typically sparse and noisy in adverse conditions. Our method achieves state-of-the-art panoptic and semantic segmentation performance on the challenging MUSES and DeLiVER datasets. Code and models will be available at https://github.com/timbroed/DGFusion

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
