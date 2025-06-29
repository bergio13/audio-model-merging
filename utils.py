"""
General utility functions for model parameter handling and compatibility checks.
"""

import torch
from typing import Dict, Any, List
import warnings

def print_model_parameters(model: torch.nn.Module, name: str):
    """Prints the shapes of parameters for a given model."""
    print(f"--- Parameters for {name} ---")
    for param_name, param in model.named_parameters():
        print(f"{param_name}: {param.shape}")

def filter_keys(state_dict: Dict[str, Any], excluded_layers: List[int] = None, additional_excluded_prefixes: List[str] = None) -> Dict[str, Any]:
    """
    Filters keys from a state dictionary based on specified prefixes and layer indices.
    Used to exclude classification heads, layer weights, or specific encoder layers.
    """
    if excluded_layers is None:
        excluded_layers = []
    if additional_excluded_prefixes is None:
        additional_excluded_prefixes = []

    keys_to_exclude_prefixes = [
        "projector.",
        "classifier.",
        "layer_weights",
    ] + [f"wav2vec2.encoder.layers.{i}." for i in excluded_layers] + additional_excluded_prefixes

    filtered_dict = {}
    for k, v in state_dict.items():
        if not any(k.startswith(prefix) for prefix in keys_to_exclude_prefixes):
            filtered_dict[k] = v
    return filtered_dict

def strip_prefix(state_dict: Dict[str, Any], prefix: str = None) -> Dict[str, Any]:
    """Strips a given prefix from the keys of a state dictionary."""
    if prefix:
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


def check_model_compatibility(models_dict: Dict[str, Any]):
    """Checks if model classes are consistent and if filtered state dict keys match."""
    print("\n--- Model Compatibility Checks ---")
    model1 = models_dict["model_wav2vec2_ks"]
    model2 = models_dict["model_wav2vec2_langid"]
    model3 = models_dict["model_wav2vec2_sentiment"]
    model4 = models_dict["model_wav2vec2_sv"]
    base_model = models_dict["model_wav2vec2_base"]

    # Check if all Wav2Vec2 models are of the same class
    if not all(m.__class__ == model1.__class__ for m in [model2, model3, model4]):
        raise ValueError("Wav2Vec2 model classes differ.")

    # Uncomment and adapt this if you need to compare specific weights
    # def compare_weights_internal(model_a, model_b):
    #     model_a_params = {name: param for name, param in model_a.named_parameters() if 'classifier' not in name and "layer_weights" not in name and "feature_extractor" not in name}
    #     model_b_params = {name: param for name, param in model_b.named_parameters() if 'classifier' not in name and "layer_weights" not in name and "feature_extractor" not in name}
    #     for (name_a, param_a), (name_b, param_b) in zip(model_a_params.items(), model_b_params.items()):
    #         diff = (param_a - param_b).abs()
    #         if diff.mean().item() == 0:
    #             print(f"Layer: {name_a}, Max difference: {diff.max().item()}, Mean difference: {diff.mean().item()}")
    #             raise ValueError("Difference equal to zero")
    #     print(f"Weights compared for {model_a.config._name_or_path} and {model_b.config._name_or_path}. No zero difference found.")
    # compare_weights_internal(model1, model2)

    # Check consistency of filtered state dictionary keys and shapes
    filtered_state_dict_1 = filter_keys(model1.state_dict())
    filtered_state_dict_2 = filter_keys(model2.state_dict())
    filtered_state_dict_3 = filter_keys(model3.state_dict())
    filtered_state_dict_4 = filter_keys(model4.state_dict())

    all_filtered_keys = [
        set(filtered_state_dict_1.keys()),
        set(filtered_state_dict_2.keys()),
        set(filtered_state_dict_3.keys()),
        set(filtered_state_dict_4.keys())
    ]

    # Check if all sets of keys are identical
    if not all(k == all_filtered_keys[0] for k in all_filtered_keys):
        raise ValueError("Filtered state dict keys mismatch across Wav2Vec2 models.")

    # Check if corresponding parameter shapes are identical
    for key in filtered_state_dict_1.keys():
        if key.startswith("classifier"):
            continue # Classifier heads can differ
        if not (
            filtered_state_dict_1[key].shape == filtered_state_dict_2[key].shape and
            filtered_state_dict_1[key].shape == filtered_state_dict_3[key].shape and
            filtered_state_dict_1[key].shape == filtered_state_dict_4[key].shape
        ):
            raise ValueError(f"Shape mismatch for key '{key}' across Wav2Vec2 models.")

    print("Filtered state dict keys and shapes match across Wav2Vec2 models.")