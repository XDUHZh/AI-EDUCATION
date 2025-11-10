def calculate_accuracy(self, test_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    category_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_questions = len(test_responses)

    for response in test_responses:
        category = response["category"]
        category_scores[category]["total"] += 1
        if response["correct_answer"] == response["model_prediction"]:
            category_scores[category]["correct"] += 1
            total_correct += 1

    category_accuracy = {
        category: scores["correct"] / scores["total"]
        for category, scores in category_scores.items()
    }
    overall_accuracy = total_correct / total_questions

    return {
        "category_accuracy": category_accuracy,
        "overall_accuracy": overall_accuracy,
    }