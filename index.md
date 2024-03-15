---
title: Improving Performance of Vision Encoding Large Language Models with Contextual Prompts
feature_text: |
feature_image: "https://bairesdev.mo.cloudinary.net/blog/2023/09/AI-In-Healthcare.jpg?tx=w_1920,q_auto"
excerpt: "An exploration and navigation of using state-of-the-art Deep Learning and Large Language models to interpret X-Ray images and generate reports."
---

## Overview

This is a a Data Science Capstone Project investigated and put together by **Luke Taylor**, **Muchan Li**, and **Raymond Song** under the mentorship of ***Albert Hsiao, MD, PhD***. All are affiliated with the Halicioglu Data Science Institute at UC San Diego. Professor Hsiao is additionally affiliated with UC San Diego Health and Radiology Department.

<div style="text-align: center; padding: 15px">
  {% include button.html text="Codebase" icon="github" link="https://github.com/raymondsong00/Xray-Report-Generator" color="#0366d6" %}
  {% include button.html text="Report 📝" link="https://github.com/Lord-of-Bugs/LLaVA-XRay-Vision" color="#f68140" %}
  {% include button.html text="Poster 🪧" link="https://drive.google.com/file/d/11IWMHwXypiSh9SmqlCkUYjpxBJvzO2lW/view?usp=drivesdk" color="#8594e4" %}
  {% include button.html text="Team Info 👨‍💻" link="/people/" color="#3baea0" %}
</div>

## Problem Statement and Motivation

Deep learning models like convolutional neurla networks (CNNs) have demonstrated promising ways of application in automating radiograph analysis, detecting conditions such as pulmonary edema, pneumothorax, and so forth. However, vast amount of data other than the radiographs has been left out, such radiologist reports, in the modeling process and can potentially make automation and predicition more accurate and usable. Inspired by advancements in Large Language Models (LLMs) and Vision Transformers (ViTs), we’re exploring multi-modal models that integrate text and image data for improved results.

![Multi-modal model](./imgs/image3.gif)

## Objectives

1. Navigate how multi-modal models are used to leverage both X-Ray reports and images to generate reports.
2. Assess whether the generated reports mimic the style of reports written by the expert readers.
3. Assess the accuracy of the generated reports in identifying pathologies.
   1. **Determine whether additional context in the text input of an LLM improves generated text outcomes**.
4. Explore the effect of prompt engineering on report generation and accuracy.
   1. **Evaluate and quantify the improvement in the LLM for Chest X-rays given more context**.

## Methods

- About 100K Chest Radiographs and their text reports from UCSD Health
  - Among which, about 97000 images are used to fine-tune the model, and about 2000 images are reserved as the test set.
- Fine-tuned a Large Language Model (LLM) with vision capabilities instead of training both a vision CNN tower and a LLM from scratch
- Using Large Language and Vision Assistant v1.5 (LLaVA, see [1][2][3]) as the base model which is based on Vicuna 13B v1.5 and CLIP ViT-L/14 visual encoder
- Input is an X-ray and then a prompt that may include information about the previous clinical history and additional patient context.
- Ground truth output is actual radiologist report for the corresponding X-ray
- The fine-tuning of the model was done using a Nvidia RTX A6000 with LORA adapters due to GPU memory constraints.
- Evaluated with cosine similarity scores between bio term specific sentence transformer embeddings.
- Extracted label probabilities for common lung conditions using Facebook BART zero shot classification to evaluate diagnostic accuracy.

## Findings and Results

In this section, we will present the insights obtained from analyzing the radiograph reports and training and testing different tricks to LLaVA model. We will focus our presentation more on the latter.

### Expert Radiologists

### Similarity Between Generated Reports and Expert Ground Truths

<div style="text-align:center; width=100%;"><img src="./imgs/radiologist_findings_lengths.png" width="48%"/><img src="./imgs/llava_findings_lengths.png" width="48%"/></div>
<div style="text-align:center; width=100%;"><img src="./imgs/radiologist_impression_lengths.png" width="48%"/><img src="./imgs/llava_impression_lengths.png" width="48%"/></div>

![top-similarity](./imgs/top-10-similarity-author-medians.png)

Click on each radiologist's name below to learn model's performance with respect to each individual:
<details>
  <summary>👨‍⚕️ Dr. Seth Kligerman</summary>
  <img src="./imgs/Kligerman-similarity.png"/>
</details>
<details>
  <summary>👨‍⚕️ Dr. Lewis Hahn</summary>
  <img src="./imgs/Hahn-similarity.png"/>
</details>
<details>
  <summary>👨‍⚕️ Dr. Michael Horowitz</summary>
  <img src="./imgs/Horowitz-similarity.png"/>
</details>
<details>
  <summary>👨‍⚕️ Dr. Ravi Rajpoot</summary>
  <img src="./imgs/Rajpoot-similarity.png"/>
</details>
<details>
  <summary>👩‍⚕️ Dr. Sharon Brouha</summary>
  <img src="./imgs/Brouha-similarity.png"/>
</details>
<details>
  <summary>👩‍⚕️ Dr. Kathleen Jacobs</summary>
  <img src="./imgs/Jacobs-similarity.png"/>
</details>
<details>
  <summary>👩‍⚕️ Dr. Elizabeth Weihe</summary>
  <img src="./imgs/Weihe-similarity.png"/>
</details>
<details>
  <summary>👨‍⚕️ Dr. Albert Hsiao</summary>
  <img src="./imgs/Hsiao-similarity.png"/>
</details>
<details>
  <summary>👨‍⚕️ Dr. William Ladd</summary>
  <img src="./imgs/Ladd-similarity.png"/>
</details>
<details>
  <summary>👨‍⚕️ Dr. Andrew Yen</summary>
  <img src="./imgs/Yen-similarity.png">
</details>

### Prompts Constructed

![prompt](./imgs/prompt.png)

The prompt on the left is designed to be generic, similar to the one used in [4] for querying model responses. It serves as our benchmark. The prompt on the right is more context-embedded and -specific. It's design is largely inspired by the spirits of chain-of-thought prompting proposed in [5]. We aim to embed more patient history and x-ray context in prompting the model to give a more holistic reading of the x-ray, treating as if it is a real diagnostic expert.

---

### Pathology Detection Outcomes

#### X-Rays and Ground Truths

<div style="padding-top: 10px; text-align:center; width=100%;">
  <img src="./imgs/Hofupo_53628087.jpg" width="48%"/><img src="./imgs/Dietepscat_53989429.jpg" width="48%"/>
  <figcaption>Pathology Present (Left) vs. Normal (Right) Chest X-Ray</figcaption>
</div>

![ground-truth](./imgs/ground-truth-report.png)

---

#### Model Generated Reports

![llava-response](./imgs/llava-response.png)

One can see that when LLaVA is prompted with the baseline generic prompt, the readings responses to both X-rays, regardless of pathological entities, are simply direct copies of each other. This indicates that either LLaVA has not learned any X-ray specific imaging feature and just memorized the report text (since the prompt are first fed to the model via fine-tuning), or that the prompt is not specific enough to elicit reasoning.

Then, when we prompted LLaVA with the context-embedded prompt, we can see that LLaVA instead can articulate distinct findings with regard to each X-ray. Specifically, since we provided the corresponding radiologist reader, the model did better in memorizing and mimicking reader-specific reporting style. Moreover, the seemingly increased pathology detection accuracy--as an example, **LLaVA correctly identified interval removal of a right IJ (sheath)**--suggests that model is now better at comprehending the task and retrieving the relevant vision and language features.

---

#### Quantified Outcomes

<div style="padding-top: 10px; text-align:center; width=100%;"><img src="./imgs/roc_curves_final_generic2.png" width="48%"/><img src="./imgs/roc_curves_final.png" width="48%"/></div>

According to the ROC plot on the left, LLaVA's accuracy in detecting pathological entities when prompted using the generic prompt is no better than random chance (represented by the dotted diagonal line), with the highest area under the curve (AUC) at 0.53 for detecting cardiomegaly in its generated reports.

According to the ROC plot on the right, LLaVA's accuracy in detecting pathological entities when prompted using the context-embedded prompt has increased, with AUC for all pathological entities placed above 0.50. The best detection accuracy LLaVA's presented is in identifying pneumothorax, with AUC at 0.73. This AUC score is considered fair by conventional standards.

<div style="text-align:center; width=100%;"><img src="./imgs/report_similarity_generic_2.png" width="48%"/><img src="./imgs/report_similarity_context_embedded.png" width="48%"/></div>

The histogram on the left represents the test-set cosine similarity scores between the semantic embeddings of LLaVA generated reports according to the **genetic prompt** and corresponding ground truth reports. The distribution is approximately normal, centered at 0.52.

The histogram on the left represents the test-set cosine similarity scores between the semantic embeddings of LLaVA generated reports according to the **context-embedded prompt** and corresponding ground truth reports. The distribution is more left-skewed, with a diminishing tail in the low-scoring regions. It is now centered at 0.63, indicating that the responses are now more semantically similar to ground truths compared to the ones before, which means LLaVA has indeed learned to mimic the reporting of each radiologist more closely.

![confusion-matrix](./imgs/confusion_matrices.png)

## Future Directions

## References

[1] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.
Improved baselines with visual instruction tuning, 2023.

[2] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and
Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.

[3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning, 2023.

[4] Sun, Yuxuan, et al. "Pathasst: Redefining pathology through generative foundation ai assistant for pathology." arXiv preprint arXiv:2305.15072 (2023).

[5] Wei, Jason, et al. "Chain-of-thought prompting elicits reasoning in large language models." Advances in neural information processing systems 35 (2022): 24824-24837.
