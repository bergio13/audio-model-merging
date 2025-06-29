"""
Functions for building Hugging Face pipelines from merged model state dictionaries.
"""

import torch
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    pipeline,
    HubertForSequenceClassification,
    AutoFeatureExtractor
)
from typing import Dict, Any, List
import warnings

def build_merged_pipeline(
    reference_model: Wav2Vec2ForSequenceClassification,
    merged_state_dict: Dict[str, Any],
    excluded_layers: List[int] = None,
    additional_excluded_prefixes: List[str] = None,
    base_model_name: str = "facebook/wav2vec2-base"
) -> pipeline:
    """Builds a Hugging Face pipeline from a merged Wav2Vec2 model state dictionary."""
    if excluded_layers is None:
        excluded_layers = []
    if additional_excluded_prefixes is None:
        additional_excluded_prefixes = []

    config = reference_model.config
    new_model = Wav2Vec2ForSequenceClassification(config)
    model_state_dict = new_model.state_dict()

    for key, val in merged_state_dict.items():
        if key in model_state_dict:
            model_state_dict[key] = val.to(new_model.device) # Ensure tensor is on correct device
        else:
            warnings.warn(f"Merged key '{key}' not found in new model. Skipping.")
    new_model.load_state_dict(model_state_dict, strict=False)

    # Restore excluded layers and other components from reference model if specified
    for layer_idx in excluded_layers:
        if 0 <= layer_idx < len(reference_model.wav2vec2.encoder.layers):
            new_model.wav2vec2.encoder.layers[layer_idx].load_state_dict(
                reference_model.wav2vec2.encoder.layers[layer_idx].state_dict()
            )
            print(f"Restored wav2vec2.encoder.layers[{layer_idx}] from reference model.")

    if any(p.startswith("wav2vec2.masked_spec_embed") for p in additional_excluded_prefixes):
        if hasattr(new_model.wav2vec2, 'masked_spec_embed') and hasattr(reference_model.wav2vec2, 'masked_spec_embed'):
            new_model.wav2vec2.masked_spec_embed.copy_(reference_model.wav2vec2.masked_spec_embed)
            print("Copied wav2vec2.masked_spec_embed from reference model.")

    if any(p.startswith("wav2vec2.feature_extractor") for p in additional_excluded_prefixes):
        if hasattr(new_model.wav2vec2, 'feature_extractor') and hasattr(reference_model.wav2vec2, 'feature_extractor'):
            for i in range(len(reference_model.wav2vec2.feature_extractor.conv_layers)):
                new_model.wav2vec2.feature_extractor.conv_layers[i].load_state_dict(
                    reference_model.wav2vec2.feature_extractor.conv_layers[i].state_dict()
                )
            print("Copied wav2vec2.feature_extractor from reference model.")

    if any(p.startswith("wav2vec2.feature_projection") for p in additional_excluded_prefixes):
        if hasattr(new_model.wav2vec2, 'feature_projection') and hasattr(reference_model.wav2vec2, 'feature_projection'):
            new_model.wav2vec2.feature_projection.load_state_dict(reference_model.wav2vec2.feature_projection.state_dict())
            print("Copied wav2vec2.feature_projection from reference model.")

    if any(p.startswith("wav2vec2.encoder.pos_conv_embed") for p in additional_excluded_prefixes):
        if hasattr(new_model.wav2vec2.encoder, 'pos_conv_embed') and hasattr(reference_model.wav2vec2.encoder, 'pos_conv_embed'):
            new_model.wav2vec2.encoder.pos_conv_embed.load_state_dict(reference_model.wav2vec2.encoder.pos_conv_embed.state_dict())
            print("Copied wav2vec2.encoder.pos_conv_embed from reference model.")

    if any(p.startswith("wav2vec2.encoder.layer_norm") for p in additional_excluded_prefixes):
        if hasattr(new_model.wav2vec2.encoder, 'layer_norm') and hasattr(reference_model.wav2vec2.encoder, 'layer_norm'):
            new_model.wav2vec2.encoder.layer_norm.load_state_dict(reference_model.wav2vec2.encoder.layer_norm.state_dict())
            print("Copied wav2vec2.encoder.layer_norm from reference model.")

    # Copy task-specific 'projector' and 'classifier' heads
    if hasattr(new_model, 'projector') and hasattr(reference_model, 'projector'):
        new_model.projector.load_state_dict(reference_model.projector.state_dict())
    if hasattr(new_model, 'classifier') and hasattr(reference_model, 'classifier'):
        new_model.classifier.load_state_dict(reference_model.classifier.state_dict())

    # Copy label mappings
    new_model.config.label2id = reference_model.config.label2id
    new_model.config.id2label = reference_model.config.id2label

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_name)
    return pipeline("audio-classification", model=new_model, feature_extractor=feature_extractor, device=new_model.device)

def build_merged_pipeline_hubert(
    reference_model: HubertForSequenceClassification,
    merged_state_dict: Dict[str, Any],
    base_model_name: str = "facebook/hubert-base-ls960"
) -> pipeline:
    """Builds a Hugging Face pipeline from a merged HuBERT model state dictionary."""
    config = reference_model.config
    new_model = HubertForSequenceClassification(config)
    model_state_dict = new_model.state_dict()

    for key, val in merged_state_dict.items():
        if key in model_state_dict:
            model_state_dict[key] = val.to(new_model.device) # Ensure tensor is on correct device
        else:
            warnings.warn(f"Key '{key}' not found in new HuBERT model. Skipping.")
    new_model.load_state_dict(model_state_dict, strict=False)

    # Copy projector and classifier heads if present
    if hasattr(new_model, 'projector') and hasattr(reference_model, 'projector'):
        new_model.projector.load_state_dict(reference_model.projector.state_dict())
    if hasattr(new_model, 'classifier') and hasattr(reference_model, 'classifier'):
        new_model.classifier.load_state_dict(reference_model.classifier.state_dict())
    if hasattr(new_model, 'feature_extractor') and hasattr(reference_model, 'feature_extractor'):
        new_model.feature_extractor.load_state_dict(reference_model.feature_extractor.state_dict())
    if hasattr(new_model, 'feature_projection') and hasattr(reference_model, 'feature_projection'):
        new_model.feature_projection.load_state_dict(reference_model.feature_projection.state_dict())

    # Copy label mappings
    new_model.config.label2id = reference_model.config.label2id
    new_model.config.id2label = reference_model.config.id2label

    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)
    return pipeline("audio-classification", model=new_model, feature_extractor=feature_extractor, device=new_model.device)