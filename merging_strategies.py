"""
Functions for implementing different model merging strategies.
"""

import torch
from typing import Dict, Any, List
import warnings
from utils import strip_prefix

def iso_merge(weights: Dict[str, float], filtered_state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Performs isotropic merging of model state dictionaries."""
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        warnings.warn("Weights do not sum to 1. Normalizing...")
        weights = {k: v / total_weight for k, v in weights.items()}

    merged_state_dict_weighted = {}
    if not filtered_state_dicts:
        return merged_state_dict_weighted # Return empty if no state dicts

    keys = next(iter(filtered_state_dicts)).keys() # Get keys from the first dict
    for key in keys:
        merged_state_dict_weighted[key] = sum(
            [weights[f'model{i+1}'] * filtered_state_dicts[i][key] for i in range(len(filtered_state_dicts))]
        )
    return merged_state_dict_weighted

def compute_task_vector(model_state_dict: Dict[str, Any], base_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Computes the task vector by subtracting base model weights from task-specific model weights."""
    return {
        k: model_state_dict[k] - base_state_dict[k]
        for k in model_state_dict
        if not k.startswith("classifier") and k in base_state_dict
    }

def task_arithmetic(
    base_state_dict: Dict[str, Any],
    model_state_dicts: List[Dict[str, Any]],
    coeffs: List[float],
    prefix: str = ""
) -> Dict[str, Any]:
    """Performs Task Arithmetic merging of model state dictionaries."""
    assert len(model_state_dicts) == len(coeffs), "Each model must have a corresponding coefficient"

    base = strip_prefix(base_state_dict, prefix)
    models_stripped = [strip_prefix(sd, prefix) for sd in model_state_dicts]

    task_vectors = []
    for idx, m_sd in enumerate(models_stripped):
        vec = compute_task_vector(m_sd, base)
        task_vectors.append(vec)

    merged_state_dict = {}
    for k in base:
        if k.startswith("classifier"):
            continue # Skip classifier heads for task arithmetic
        merged_value = base[k].clone()
        for i, vec in enumerate(task_vectors):
            if k in vec: # Ensure key exists in task vector
                merged_value += coeffs[i] * vec[k]
        merged_state_dict[prefix + k] = merged_value

    return merged_state_dict

def apply_dare_mask(task_vector: Dict[str, Any], p: float = 0.1) -> Dict[str, Any]:
    """Performs a DARE mask to a task vector."""
    masked_vector = {}
    for k, v in task_vector.items():
        mask = torch.bernoulli(torch.full_like(v, 1 - p))
        masked_vector[k] = (v * mask) / (1 - p)
    return masked_vector

def dare(base_state_dict: Dict[str, Any], model_state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Performs DARE (Dropout-Aware Rescaling) merging of model state dictionaries."""
    merged_state_dict = {}
    task_vectors = []
    for model_sd in model_state_dicts:
        task_vector = compute_task_vector(model_sd, base_state_dict)
        task_vector = apply_dare_mask(task_vector) # Apply DARE mask
        task_vectors.append(task_vector)

    keys = [k for k in base_state_dict if not k.startswith("classifier")] # Exclude classifiers
    for k in keys:
        merged_value = base_state_dict[k]
        for i, vec in enumerate(task_vectors):
            # Add masked task vector to base model
            if k in vec: # Ensure key exists in task vector
                merged_value = merged_value + vec[k]
        merged_state_dict[k] = merged_value
    return merged_state_dict