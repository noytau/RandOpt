"""Reward scoring for ImageNet-C image classification.

Responses are predicted class labels (the 0..999 ImageNet class index as a
string). Kept in the standard reward-module shape so LLM-style responses
("<answer>17</answer>" or free text ending in a label) also score correctly.
"""
import re


def extract_answer(response: str) -> str:
    """Pull the predicted label out of the model's response."""
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip().split("\n")[-1].strip()


def compute_score(response: str, ground_truth: str) -> float:
    """Return 1.0 if the predicted label matches, 0.0 otherwise."""
    ans = extract_answer(response)
    return 1.0 if ans.strip().lower() == str(ground_truth).strip().lower() else 0.0
