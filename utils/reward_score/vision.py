def compute_score(predicted_class_id: int, ground_truth_class_id: int) -> float:
    return 1.0 if predicted_class_id == ground_truth_class_id else 0.0
