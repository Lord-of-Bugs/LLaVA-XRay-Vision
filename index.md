---
title: Improving Performance of Vision Encoding Large Language Models with Contextual Prompts
feature_text: |
feature_image: "https://bairesdev.mo.cloudinary.net/blog/2023/09/AI-In-Healthcare.jpg?tx=w_1920,q_auto"
excerpt: "An exploration and navigation of using state-of-the-art Deep Learning and Large Language models to interpret X-Ray images and generate reports."
---

## Overview

This is a a Data Science Capstone Project investigated and put together by **Luke Taylor**, **Muchan Li**, and **Raymond Song** under the mentorship of ***Albert Hsiao, MD, PhD***. All are affiliated with the Halicioglu Data Science Institute at UC San Diego. Professor Hsiao is additionally affiliated with UC San Diego Health and Radiology Department.

<div style="text-align: center; padding: 15px">
  {% include button.html text="Codebase" icon="github" link="https://github.com/Lord-of-Bugs/LLaVA-XRay-Vision" color="#0366d6" %}
  {% include button.html text="Report 📝" link="https://github.com/Lord-of-Bugs/LLaVA-XRay-Vision" color="#f68140" %}
  {% include button.html text="Poster 🪧" link="https://github.com/Lord-of-Bugs/LLaVA-XRay-Vision" color="#8594e4" %}
  {% include button.html text="Team Info 👨‍💻" link="/people/" color="#3baea0" %}
</div>

## Problem Statement and Motivation

Deep learning models like convolutional neurla networks (CNNs) have demonstrated promising ways of application in automating radiograph analysis, detecting conditions such as pulmonary edema, pneumothorax, and so forth. However, vast amount of data other than the radiographs has been left out, such radiologist reports, in the modeling process and can potentially make automation and predicition more accurate and usable. Inspired by advancements in Large Language Models (LLMs) and Vision Transformers (ViTs), we’re exploring multi-modal models that integrate text and image data for improved results.

## Objectives

1. Navigate how multi-modal models are used to leverage both X-Ray reports and images to generate reports.
2. Assess whether the generated reports mimic the style of reports written by the expert readers.
3. Assess the accuracy of the generated reports in identifying pathologies.
   1. **Determine whether additional context in the text input of an LLM improves generated text outcomes**.
4. Explore the effect of prompt engineering on report generation and accuracy.
   1. **Evaluate and quantify the improvement in the LLM for Chest X-rays given more context**.

## Methods

- About 100K Chest Radiographs and their text reports from UCSD Health
- Fine-tuned a Large Language Model (LLM) with vision capabilities instead of training both a vision CNN tower and a LLM from scratch
- Using Large Language and Vision Assistant v1.5 (LLaVA)[1][2][3] as the base model which is based on Vicuna 13B v1.5 and CLIP ViT-L/14 visual encoder
- Input is an X-ray and then a prompt that may include information about the previous clinical history and additional patient context.
- Ground truth output is actual radiologist report for the corresponding X-ray
- The fine-tuning of the model was done using a Nvidia RTX A6000 with LORA adapters due to GPU memory constraints.
- Evaluated with cosine similarity scores between bio term specific sentence transformer embeddings.
- Extracted label probabilities for common lung conditions using Facebook BART zero shot classification to evaluate diagnostic accuracy.

## Results and Analysis

## Future Directions

## References
