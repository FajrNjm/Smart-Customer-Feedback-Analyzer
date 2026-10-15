# Smart Customer Feedback Analyzer

### Applied NLP Engineering — Capstone Project

Developed as an individual capstone project for the
AI Engineering Track — Applied NLP Engineering Bootcamp.

**SDAIA Academy:** https://github.com/SDAIAAcademy

## 1. Problem

Organizations receive large amounts of customer feedback
in different languages. Manually reviewing this feedback
can be time-consuming.

This project demonstrates an NLP prototype that:

- Classifies Arabic and English text sentiment.
- Explores semantic clusters using text embeddings.
- Summarizes observed cluster statistics.
- Uses a generative LLM with basic output validation.

## 2. Data

Dataset:
https://huggingface.co/datasets/clapAI/MultiLingualSentiment

The prepared sample contains:

| Split | Arabic | English | Total |
|---|---:|---:|---:|
| Train | 75 | 75 | 150 |
| Validation | 24 | 24 | 48 |
| Test | 24 | 24 | 48 |

Classes: Positive, Neutral, Negative.

The sample is balanced by language and sentiment.

Limitation: The source contains multiple domains,
including customer reviews, book reviews, and social
media posts. It is not exclusively customer feedback.

## 3. Approach

### Sentiment Classification

Baseline:
Word TF-IDF + Logistic Regression.

Alternative experiments:
- Multilingual Sentence Embeddings + Logistic Regression
- Character-level TF-IDF + Logistic Regression
- Combined Word and Character TF-IDF

The model with the highest validation Macro F1
was selected for the final evaluation.

### Semantic Clustering

Model:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Sentence embeddings are used with K-Means to explore
similar feedback.

Clustering is performed separately for Arabic and English
on a filtered training-data subset.

The application displays precomputed cluster statistics.
It does not assign new user reviews to clusters live.

### Generative LLM

Model: Qwen/Qwen2.5-1.5B-Instruct

Library: Hugging Face Transformers.

The LLM receives Python-computed cluster statistics
and generates a natural-language summary.

Basic checks examine the generated output.
When validation fails or the LLM is unavailable,
a deterministic Python summary is displayed.

This validation is limited and does not guarantee
that all possible hallucinations will be detected.

### Fine-Tuning

No Transformer fine-tuning was performed.

The sentiment classifier was trained on task-specific data.
The embedding model and the LLM use pretrained weights.

## 4. Results

### Validation Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Word TF-IDF baseline | 50.00% | 49.15% |
| Multilingual embeddings | 41.67% | 40.01% |
| Character TF-IDF | 45.83% | 46.30% |
| Combined TF-IDF | 45.83% | 45.80% |

The original baseline achieved the highest validation
Macro F1 among the tested configurations.

Alternative models did not improve the validation score.

### Final Held-Out Test Results

- Accuracy: 50.00%
- Macro F1: 48.48%
- Correct predictions: 24 of 48

| Language | Accuracy | Macro F1 |
|---|---:|---:|
| Arabic | 37.50% | 35.87% |
| English | 62.50% | 60.89% |

The test sample is small, so these results should not
be generalized to larger real-world populations.

### Error Analysis

24 incorrect predictions were observed on the final
test set.

10 cases were manually reviewed and assigned
possible failure reasons, including:

- Dialectal Arabic
- Mixed sentiment
- Negation
- Label ambiguity
- Domain mismatch and non-review text
- Long or complex reviews

See results/error_analysis_public.csv.

### Clustering Insights

Four exploratory groups were created across the two
languages.

Cluster ar_1 had the highest observed negative
review rate (56.2%).

Cluster en_0 had the lowest (22.2%).

These rates come from original dataset labels,
not predictions on new customer feedback.

The clusters have not been validated as specific
customer issue categories.

## 5. How to Run

### Requirements

Python and the packages in requirements.txt.

A CUDA-compatible GPU is recommended for local
Qwen inference. CPU-only operation supports the
sentiment classifier and saved insights. The LLM
summary uses a deterministic fallback without a GPU.

### Setup

Clone this repository and enter the project folder:

```bash
git clone https://github.com/FajrNjm/Smart-Customer-Feedback-Analyzer
cd Smart-Customer-Feedback-Analyzer
pip install -r requirements.txt
python app.py
```

Replace YOUR_REPOSITORY_URL with the actual repository
URL after uploading the project.

Open the local Gradio URL shown in the terminal.

The first LLM request may take longer because its
pretrained model weights must be downloaded.

### Google Colab

The project notebook documents data preparation,
training, model comparison, clustering, and evaluation.

## 6. Sample Inputs and Outputs

Example 1:

Input: The customer service was terrible.

Expected sentiment: Negative.

Example 2:

Input: الخدمة ممتازة والتعامل رائع

Expected sentiment: Positive.

These are illustrative expected sentiment labels.
The model can make errors, and probability scores
should not be interpreted as calibrated certainty.

The Customer Insights tab displays saved cluster
statistics and offers LLM-based summary generation.

## 7. Evaluation Files

The results directory includes:

- confusion_matrix.png
- language_evaluation.csv
- validation_by_language.csv
- model_comparison_by_language.csv
- error_analysis_public.csv
- customer_insights.csv
- insights_conclusion.txt
- llm_raw_summary.txt
- llm_verified_summary.txt

## 8. Limitations

- Small labeled training sample (150 records).
- Mixed source domains.
- Lower Arabic test performance.
- Weak exploratory cluster separation.
- Limited generalization to unseen customer feedback.
- Model probabilities are not calibrated.
- LLM output validation is rule-based and incomplete.
- GPU resources may be needed for live LLM inference.

## 9. Reproducibility

The repository provides source code, prepared artifacts,
model-selection results, and evaluation outputs.

Results were recorded from controlled train,
validation, and test splits.

Do not use the held-out test set to select new models.

## 10. Acknowledgments

This project was developed for the Applied NLP
Engineering Bootcamp.

SDAIA Academy:
https://github.com/SDAIAAcademy

Hugging Face:
https://huggingface.co/

The project is an individual educational prototype
and is not an official SDAIA Academy product.

## Data Redistribution Note

Raw review text and prepared dataset CSV files are not included in this repository pending verification of redistribution permissions. The public error analysis contains only test row indices, manually assigned failure categories, and tentative explanations.
