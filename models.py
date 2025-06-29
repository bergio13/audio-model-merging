"""
Functions for loading and initializing Hugging Face models and pipelines.
"""

import torch
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    pipeline,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoProcessor,
    AutoModel,
    HubertForSequenceClassification
)
from typing import Dict, Any

def load_and_initialize_models(device: torch.device) -> Dict[str, Any]:
    """Loads and initializes all required models and pipelines."""
    print("\n--- Model Loading ---")

    # HuBERT models
    model_hubert_base = AutoModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    pipe_hubert_ks = pipeline("audio-classification", model="Graphcore/hubert-base-superb-ks", device=device)
    pipe_hubert_langid = pipeline("audio-classification", model="Graphcore/hubert-base-common-language", device=device)

    model_hubert_ks = pipe_hubert_ks.model
    model_hubert_langid = pipe_hubert_langid.model

    # Wav2Vec2 pipelines
    pipe_wav2vec2_ks = pipeline("audio-classification", model="anton-l/wav2vec2-base-ft-keyword-spotting", device=device)
    pipe_wav2vec2_langid = pipeline("audio-classification", model="anton-l/wav2vec2-base-lang-id", device=device)
    pipe_wav2vec2_sentiment = pipeline("audio-classification", model="somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-classification-MESD", device=device)

    # Load Wav2Vec2 model directly
    model_wav2vec2_sv = Wav2Vec2ForSequenceClassification.from_pretrained(
        "anton-l/wav2vec2-base-superb-sv",
        ignore_mismatched_sizes=True
    ).to(device)

    # Extract models from pipelines
    model_wav2vec2_ks = pipe_wav2vec2_ks.model
    model_wav2vec2_langid = pipe_wav2vec2_langid.model
    model_wav2vec2_sentiment = pipe_wav2vec2_sentiment.model

    # Base Wav2Vec2 model
    model_wav2vec2_base = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        ignore_mismatched_sizes=True
    ).to(device)

    return {
        "model_wav2vec2_base": model_wav2vec2_base,
        "model_wav2vec2_ks": model_wav2vec2_ks,
        "model_wav2vec2_langid": model_wav2vec2_langid,
        "model_wav2vec2_sentiment": model_wav2vec2_sentiment,
        "model_wav2vec2_sv": model_wav2vec2_sv,
        "model_hubert_base": model_hubert_base,
        "model_hubert_ks": model_hubert_ks,
        "model_hubert_langid": model_hubert_langid,
        "pipe_hubert_ks": pipe_hubert_ks,
        "pipe_hubert_langid": pipe_hubert_langid,
        "pipe_wav2vec2_ks": pipe_wav2vec2_ks, 
        "pipe_wav2vec2_langid": pipe_wav2vec2_langid, 
        "pipe_wav2vec2_sentiment": pipe_wav2vec2_sentiment
    }