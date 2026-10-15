
import re
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

model = joblib.load(
    ROOT / "models" / "sentiment_classifier.joblib"
)

insights = pd.read_csv(
    ROOT / "results" / "customer_insights.csv"
)

_llm = None
_tokenizer = None


def predict_review(text):
    if not text or not text.strip():
        return "", {}, "Please enter a review."

    prediction = model.predict([text])[0]
    probs = model.predict_proba([text])[0]

    scores = {
        str(label): float(value)
        for label, value in zip(model.classes_, probs)
    }

    top = sorted(probs, reverse=True)
    margin = float(top[0] - top[1])

    note = (
        "Uncertain prediction: probability scores "
        "are close. Interpret with caution."
        if margin < 0.10 else
        "Model probability estimates are not "
        "guaranteed confidence."
    )

    return str(prediction).capitalize(), scores, note


def show_insights():
    return insights[
        ["Cluster", "Language", "Total Reviews",
         "Positive", "Neutral", "Negative",
         "Negative Rate (%)"]
    ]


def generate_summary():
    global _llm, _tokenizer

    high = insights.loc[
        insights["Negative Rate (%)"].idxmax()
    ]
    low = insights.loc[
        insights["Negative Rate (%)"].idxmin()
    ]

    high_id, low_id = str(high["Cluster"]), str(low["Cluster"])
    high_rate = f'{high["Negative Rate (%)"]:.1f}'
    low_rate = f'{low["Negative Rate (%)"]:.1f}'

    fallback = (
        f"Cluster {high_id} has the highest observed "
        f"negative review rate ({high_rate}%), "
        f"while cluster {low_id} has the lowest "
        f"({low_rate}%). These statistics are based "
        "on labeled training reviews. Clusters are "
        "exploratory, not validated issue categories."
    )

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if _llm is None:
            if not torch.cuda.is_available():
                return fallback, (
                    "LLM unavailable: GPU required. "
                    "Python fallback displayed."
                )

            name = "Qwen/Qwen2.5-1.5B-Instruct"
            _tokenizer = AutoTokenizer.from_pretrained(name)
            _llm = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.float16, device_map="auto"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite verified facts only. "
                    "Keep IDs and numbers exact. "
                    "Do not infer customer satisfaction, "
                    "trends, causes or specific issues."
                )
            },
            {"role": "user", "content": fallback}
        ]

        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True
        )

        inputs = _tokenizer(
            prompt, return_tensors="pt"
        ).to(_llm.device)

        with torch.inference_mode():
            output = _llm.generate(
                **inputs,
                max_new_tokens=130,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id
            )

        raw = _tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        def has_pair(cluster, rate):
            return bool(re.search(
                rf'\b{re.escape(cluster)}\b'
                rf'[^.!?\n]{{0,100}}'
                rf'\b{re.escape(rate)}%',
                raw, re.I
            ))

        found_ids = set(re.findall(
            r'\b(?:ar|en)_\d+\b', raw
        ))
        found_rates = set(re.findall(
            r'\b\d+(?:\.\d+)?%', raw
        ))

        banned = [
            "most dissatisfied customers",
            "least dissatisfied customers",
            "caused by",
            "proves that",
            "customer satisfaction improved"
        ]

        lower = raw.lower()
        numeric_ok = (
            found_ids == {high_id, low_id}
            and found_rates == {
                high_rate + "%", low_rate + "%"
            }
            and has_pair(high_id, high_rate)
            and has_pair(low_id, low_rate)
        )

        relations_ok = (
            0 <= lower.find("highest") < raw.find(high_id)
            and 0 <= lower.find("lowest") < raw.find(low_id)
        )

        if (
            numeric_ok and relations_ok
            and not any(p in lower for p in banned)
        ):
            return raw, (
                "LLM generated — basic checks passed. "
                "Human review recommended."
            )

        return fallback, (
            "LLM generated, but validation rejected "
            "the output. Python fallback displayed."
        )

    except Exception as error:
        return fallback, (
            "LLM unavailable; Python fallback displayed. "
            f"Error type: {type(error).__name__}"
        )


with gr.Blocks(
    title="Smart Customer Feedback Analyzer"
) as demo:

    gr.Markdown("""
    # Smart Customer Feedback Analyzer
    **SDAIA Academy — Applied NLP Engineering**

    Arabic & English customer feedback prototype.

    - TF-IDF + Logistic Regression: sentiment
    - Sentence Transformers + K-Means: offline clustering
    - Qwen2.5-1.5B-Instruct: optional live LLM summary

    **Test Accuracy: 50% | Macro F1: 48.48%**

    Experimental results; not production-grade.
    """)

    with gr.Tab("Sentiment Analysis"):
        review = gr.Textbox(
            label="Customer Review", lines=4
        )
        button = gr.Button(
            "Analyze Sentiment", variant="primary"
        )
        sentiment = gr.Textbox(
            label="Predicted Sentiment"
        )
        scores = gr.Label(
            label="Model Probability Scores"
        )
        note = gr.Textbox(
            label="Prediction Reliability Note"
        )
        button.click(
            predict_review,
            inputs=review,
            outputs=[sentiment, scores, note]
        )

    with gr.Tab("Customer Insights + LLM"):
        gr.Markdown(
            "Exploratory clusters based on original "
            "training labels, not live predictions."
        )

        table = gr.Dataframe(interactive=False)
        show = gr.Button("Show Customer Insights")
        show.click(show_insights, outputs=table)

        generate = gr.Button(
            "Generate LLM Summary", variant="primary"
        )
        summary = gr.Textbox(
            label="Generated / Verified Summary", lines=5
        )
        status = gr.Textbox(
            label="LLM Validation Status"
        )
        generate.click(
            generate_summary,
            outputs=[summary, status]
        )


if __name__ == "__main__":
    demo.launch()
