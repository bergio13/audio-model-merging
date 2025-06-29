"""
Functions for evaluating model performance on various audio classification tasks.
"""

import numpy as np
from transformers import pipeline
from typing import Any, List

def evaluate_model_ks(pipeline_obj: pipeline, dataset: Any, label_feature: Any, verbose: bool = False) -> float:
    """Evaluates a model's accuracy on the Keyword Spotting dataset."""
    correct = 0
    total = len(dataset)

    print(f"\nEvaluating on Keyword Spotting dataset ({total} samples)...")
    for idx, sample in enumerate(dataset):
        audio = sample["audio"]
        true_label_str = label_feature.int2str(sample["label"])

        try:
            pred = pipeline_obj(audio["array"], sampling_rate=audio["sampling_rate"])
            pred_label_str = pred[0]["label"]

            if pred_label_str == true_label_str:
                correct += 1

            if verbose:
                print(f"[{idx+1}/{total}] True: {true_label_str}, Predicted: {pred_label_str}")

        except Exception as e:
            print(f"[{idx+1}/{total}] Error processing sample: {e}")
            continue

    accuracy = correct / total
    print(f"Final Accuracy (KS): {accuracy:.2%} on {total} samples.")
    return accuracy

def evaluate_model_langid(pipeline_obj: pipeline, dataset: Any, label_list: List[str], total_samples: int = 5000) -> float:
    """Evaluates a model's accuracy on the Language Identification dataset."""
    correct = 0
    total = min(total_samples, len(dataset))

    print(f"\nEvaluating on Language ID dataset ({total} samples)...")
    for i in range(total):
        sample = dataset[i]
        audio = sample["audio"]

        true_label_idx = sample["language"]
        true_label_str = label_list[true_label_idx]

        prediction = pipeline_obj(audio["array"], sampling_rate=16000)
        pred_label_str = prediction[0]["label"]

        if pred_label_str == true_label_str:
            correct += 1

    accuracy = correct / total
    print(f"Final Accuracy (Lang ID): {accuracy:.2%} on {total} samples.")
    return accuracy

def evaluate_model_sentiment(pipeline_obj: pipeline, dataset: Any, total_samples: int = None) -> float:
    """Evaluates a model's accuracy on the Sentiment Classification dataset."""
    correct = 0
    total = total_samples if total_samples is not None else len(dataset)

    print(f"\nEvaluating on Sentiment Classification dataset ({total} samples)...")
    for i in range(total):
        sample = dataset[i]
        audio = np.array(sample["audio_array"])

        true_label_str = sample["emotion"]
        prediction = pipeline_obj(audio)
        pred_label_str = prediction[0]["label"]

        if pred_label_str == true_label_str:
            correct += 1

    accuracy = correct / total
    print(f"Final Accuracy (Sentiment): {accuracy:.2%} on {total} samples.")
    return accuracy