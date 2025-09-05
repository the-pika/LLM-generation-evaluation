# LLM-generation-evaluation
LLM-Based Generation and Evaluation of Medical Discharge Summaries

## Overview

This repository contains the implementation of a two-stage framework for generating and evaluating medical discharge summaries using Large Language Models (LLMs).
The framework combines:

- Summary Generation: Producing multiple abstractive summaries from raw discharge notes using the LLaMA 3.1 model.
- Summary Evaluation: Assessing generated summaries with GPT-4 as an evaluator, using a structured prompt and Likert-scale scoring across multiple quality dimensions.

This work highlights how LLMs can act as both generators and evaluators to produce clinically faithful and patient-friendly summaries.


> ⚠️ **Notice**  
> This project is currently under review for publication. Please do not copy, reproduce, or redistribute any part of the code or dataset without explicit permission or without proper reference.


## Feautures

- Multi-candidate summary generation with configurable decoding parameters.
- Likert-scale evaluation with transparent scoring and explanations.
- Weighted aggregation scheme for optimal summary selection.
- JSON and CSV outputs for downstream analysis.
- Plots and statistical experiments (temperature vs. score, boxplots, correlation heatmaps).


## Research Context

This framework is part of my research on LLM and Multi-Objective Text Optimization.
It establishes a pipeline where:
- LLM-1 act as generators to create abstractive summaries.
- LLM-2 also act as evaluators to score and rank candidate outputs.

## Citation

If you use this framework in your research, please cite:

@misc{verma2025llm_summ_eval,
  author = {Deepika Verma},
  title = {LLM-Based Generation and Evaluation of Medical Discharge Summaries},
  year = {2025},
  published = {[\url{https://github.com/your-repo-link](https://github.com/the-pika/LLM-generation-evaluation)}}
}

