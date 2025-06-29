"""
Functions for loading and preprocessing datasets.
"""

from datasets import load_dataset, Audio
from typing import Dict, Any

def load_datasets() -> Dict[str, Any]:
    """Loads and preprocesses the datasets used for evaluation."""
    print("\n--- Data Loading ---")

    # Keyword Spotting (KS) dataset
    dataset_ks = load_dataset("superb", "ks", split="test", trust_remote_code=True)
    label_feature_ks = dataset_ks.features["label"]
    print(f"KS Dataset: {dataset_ks}")
    print(f"KS Labels: {label_feature_ks.names}")

    # Language ID dataset
    dataset_lang = load_dataset("speechbrain/common_language", split="test", trust_remote_code=True)
    dataset_lang = dataset_lang.cast_column("audio", Audio(sampling_rate=16000))
    label_list_lang = dataset_lang.features["language"].names
    print(f"Language ID Dataset: {dataset_lang}")
    print(f"Number of Language ID labels: {len(label_list_lang)}")

    # Sentiment dataset
    dataset_sentiment = load_dataset("somosnlp-hackathon-2022/MESD", split="test")
    print(f"Sentiment Dataset: {dataset_sentiment}")
    print(f"Number of Sentiment labels: {len(set(dataset_sentiment['emotion']))}")

    return {
        "dataset_ks": dataset_ks,
        "label_feature_ks": label_feature_ks,
        "dataset_lang": dataset_lang,
        "label_list_lang": label_list_lang,
        "dataset_sentiment": dataset_sentiment,
    }