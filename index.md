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
- Using Large Language and Vision Assistant v1.5 (LLaVA)[1][2][3] as the base model which is based on Vicuna 13B v1.5 and CLIP ViT-L/14 visual encoder
- Input is an X-ray and then a prompt that may include information about the previous clinical history and additional patient context.
- Ground truth output is actual radiologist report for the corresponding X-ray
- The fine-tuning of the model was done using a Nvidia RTX A6000 with LORA adapters due to GPU memory constraints.
- Evaluated with cosine similarity scores between bio term specific sentence transformer embeddings.
- Extracted label probabilities for common lung conditions using Facebook BART zero shot classification to evaluate diagnostic accuracy.

## Findings and Results

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

| Generic Prompt   |      Context Embedded Prompt      |
|:-:       |:-:           |
| Write a report for the given chest x-ray. It should contain a clear findings and impression section.  Be explicit for any abnormalities or normal findings. | AUTHOR: {...}, CLINICAL HISTORY: {...}, CONTEXT: {...}, AGE: {...}, SEX: {...}, Based on AUTHOR and CLINICAL HISTORY, suppose you were a radiologist on X-RAY,  could you provide a detailed report from this chest X-ray? |

### Pathology Detection Outcomes

<div style="text-align:center; width=100%;">
  <img src="./imgs/Hofupo_53628087.jpg" width="48%"/><img src="./imgs/Dietepscat_53989429.jpg" width="48%"/>
  <figcaption>Pathology Present (Left) vs. Normal (Right) Chest X-Ray</figcaption>
</div>

| Radiologist Report (Left)  |      Radiologist Report (Right)      |
|:-:       |:-:            |
| FINDINGS: See impression.  IMPRESSION: Interval removal of a right IJ Swan-Ganz catheter. Sheath remains present. Increased interstitial and alveolar pulmonary edema. Stable small bilateral pleural effusions with slight loculation in the right major fissure. | FINDINGS: Unremarkable cardiomediastinal silhouette.  No consolidation, pleural effusion or pneumothorax.  Basilar reticular fibrotic changes are noted, unchanged.  No acute osseous abnormality identified. IMPRESSION: Stable chest x-ray. No acute cardiopulmonary findings. |

<br>

| X-Ray |      Prompt to LLaVA      |  LLaVA Generated Report |
|:-:     |:-:            |:-:    |
| Left |  Generic Prompt | FINDINGS: Lines and Tubes: None. Mediastinum: The cardiomediastinal silhouette is unchanged. No lymphadenopathy is appreciated.  Lungs: The lungs are clear.  Pleura: No pneumothorax or effusion.  Bones and soft tissues: Unchanged. IMPRESSION: No acute cardiopulmonary abnormality. |
| Left |    Context Embedded Prompt   |   FINDINGS: See impression. IMPRESSION: Interval removal of a right IJ sheath. Right chest tubes remain present. Stable small right pneumothorax. Stable interstitial edema. |
| Right |  Generic Prompt | FINDINGS: Lines and Tubes: None. Mediastinum: The cardiomediastinal silhouette is unchanged. No lymphadenopathy is appreciated.  Lungs: The lungs are clear.  Pleura: No pneumothorax or effusion.  Bones and soft tissues: Unchanged. IMPRESSION: No acute cardiopulmonary abnormality. |
| Right | Context Embedded Prompt | FINDINGS: Unremarkable cardiomediastinal silhouette. No consolidation, pleural effusion or pneumothorax.  No acute osseous abnormality identified. IMPRESSION: No acute cardiopulmonary findings. |

<br>

<div style="text-align:center; width=100%;"><img src="./imgs/roc_curves_final_generic2.png" width="48%"/><img src="./imgs/roc_curves_final.png" width="48%"/></div>
<div style="text-align:center; width=100%;"><img src="./imgs/report_similarity_generic_2.png" width="48%"/><img src="./imgs/report_similarity_context_embedded.png" width="48%"/></div>

![confusion-matrix](./imgs/confusion_matrices.png)

## Future Directions

## References
