# Benchmarking explainable offensive language detection in Somali with human-annotated rationales

This repository is the official implementation of the Scientific Reports 2026 Paper with the title: [Benchmarking explainable offensive language detection in Somali with human-annotated rationales](https://www.nature.com/articles/s41598-026-53781-0) 


# Overview

Offensive language presents significant challenges on the internet and requires robust moderation. However, the efficacy of such moderation often depends on providing clear and interpretable justifications for each classification. Unfortunately, many
existing datasets lack annotated rationales, and most detection models offer limited interpretability and transparency. These limitations hinder the development of trustworthy systems and the implementation of effective content moderation strategies. In this paper, we introduce SomOffXplain, an interpretable framework for detecting offensive language in Somali, which generates
human-understandable explanations for its predictions. SomOffXplain performs span-level rationale extraction at both the word and phrase levels, enabling it to highlight text segments that support its predictions. Given that Somali is a low-resource
language, we first construct a new benchmark dataset of 10,175 samples, each annotated with human-provided rationales. We evaluate our method against five fine-tuned pre-trained models using Local Interpretable Model-Agnostic Explanations (LIME). Additionally, we adapt four large language models (LLMs) through few-shot and zero-shot prompting to assess their ability to
understand and produce rationales in Somali. Our proposed model demonstrates superiority in terms of explainability and predictive accuracy, exhibiting higher plausibility and faithfulness compared to the baselines. Furthermore, our results reveal that half of the state-of-the-art LLMs evaluated fail to generate high-quality rationales that align with human-annotated ground truth rationales, whereas LIME-based methods also prove to be weak explainers for Somali text. We believe our contributions support online safety, help prevent harassment in under-resourced language communities, enhance the trustworthiness of language models, and promote transparency in artificial intelligence systems.



>📋 Code and dataset instructions are coming soon. For urgent access requests, please contact **Abdisalam** **Badel** at fiicane121@gmail.com or 202214090105@std.uestc.edu.cn.





License
---------------------------------------------------------------------------------------------------------
MIT
