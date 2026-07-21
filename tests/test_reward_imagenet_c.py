"""Tests for utils/reward_score/imagenet_c.py — the scoring authority for
ImageNet-C runs (each RandOpt perturbation's fitness = mean of these scores).

Run:  python -m pytest tests/test_reward_imagenet_c.py -v
  or: python tests/test_reward_imagenet_c.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reward_score.imagenet_c import compute_score, extract_answer


# -- core classification scoring ------------------------------------------

def test_exact_match_scores_one():
    assert compute_score("17", "17") == 1.0


def test_mismatch_scores_zero():
    assert compute_score("3", "17") == 0.0


def test_int_ground_truth_accepted():
    # handlers store labels as str, but callers may pass the raw int
    assert compute_score("17", 17) == 1.0
    assert compute_score("3", 17) == 0.0


def test_whitespace_and_case_robust():
    assert compute_score("  17  ", "17") == 1.0
    # case-folding matters if labels are ever wnids/class names, not digits
    assert compute_score("N01440764", "n01440764") == 1.0


# -- extraction: <answer> tag path (LLM-style responses) ------------------

def test_answer_tag_extracted():
    assert extract_answer("<answer>17</answer>") == "17"
    assert compute_score("<answer>17</answer>", "17") == 1.0


def test_answer_tag_with_surrounding_text():
    resp = "Let me look closely.\n<answer> 17 </answer>\nDone."
    assert extract_answer(resp) == "17"
    assert compute_score(resp, "17") == 1.0


def test_answer_tag_multiline_content():
    assert extract_answer("<answer>\n17\n</answer>") == "17"


# -- extraction: last-line fallback (free-text responses) -----------------

def test_last_line_fallback():
    assert extract_answer("I think the class is\n17") == "17"
    assert compute_score("I think the class is\n17", "17") == 1.0


def test_label_mentioned_but_not_final_answer_scores_zero():
    # the label appearing mid-text must NOT count — only the answer position
    assert compute_score("it is not 17, rather\n3", "17") == 0.0


def test_single_line_response_is_its_own_answer():
    assert extract_answer("17") == "17"


# -- failure behavior: never crash, score 0.0 -----------------------------

def test_empty_response_scores_zero():
    assert compute_score("", "17") == 0.0


def test_garbage_response_scores_zero():
    assert compute_score("no idea, sorry!", "17") == 0.0


def test_empty_tag_scores_zero():
    assert compute_score("<answer></answer>", "17") == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
