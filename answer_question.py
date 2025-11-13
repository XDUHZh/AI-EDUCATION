def answer_question(
        self,
        category: str,
        question: str,
        interaction_history: List[Dict[str, str]],
        pre_test_results: Optional[List[Dict[str, Any]]] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    system_message = f"You are a student focusing on {category}. Analyze the question carefully, explain your thought process ({self.recommended_answer_token_limit} tokens or less) , and try to apply the concepts you've learned to solve problems. If you're unsure, express your uncertainty and explain your reasoning."

    few_shot_examples_message = ""
    if self.use_few_shot and few_shot_examples:
        few_shot_examples_message = "\n\nHere are some example questions and reasoning processes:\n"
        for example in few_shot_examples[:self.num_if_few_shots]:
            few_shot_examples_message += f"Question: {example['question']}\nReasoning: {example['cot_content']}\n\n"

    messages = [{"role": "system", "content": system_message + "\n" + few_shot_examples_message}]

    if self.include_pretest_info and pre_test_results:
        for r in pre_test_results:
            pre_test_question_text = self.format_question(
                r["question"], r["options"]
            )
            messages.append(
                {"role": "user", "content": f"Teacher: {pre_test_question_text}"}
            )
            pre_test_model_response = r["model_response"]
            messages.append(
                {"role": "assistant", "content": f"Student: {pre_test_model_response}"}
            )

    if interaction_history:
        for interaction in interaction_history:
            messages.append(
                {"role": "user", "content": f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', interaction['question'])}"}
            )
            messages.append(
                {"role": "assistant", "content": f"Student: {re.sub(r'^(Student:( )*)+', '', interaction['answer'])}"}
            )

    messages.append(
        {"role": "user",
         "content": f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', question)}\n\nYour thoughtful and detailed answer ({self.recommended_answer_token_limit} tokens or less):"}
    )

    # Initialize retry counter, when the output exceeds 80% of max_tokens and is greater than recommended_question_token_limit, a rerun of generation is required.
    retry_count = 0

    while retry_count < self.answer_retries:
        response = self.generate_response(messages)
        response_tokens = self.count_tokens(response)
        max_allowed_tokens = max(
            self.max_tokens * self.max_tokens_rerun_threshold_percentage,
            self.recommended_answer_token_limit,
        )

        if 0 < response_tokens <= max_allowed_tokens:
            return response
        elif response_tokens == 0:
            logging.warning(
                f"{self.name}'s {len(interaction_history) + 1} round answer for {pre_test_results[0]['question_id']} had 0 tokens (attempt {retry_count + 1}/{self.answer_retries}). Retrying...")
        else:
            logging.warning(
                f"{self.name}'s {len(interaction_history) + 1} round answer for {pre_test_results[0]['question_id']} had {response_tokens} tokens and exceeded {max_allowed_tokens} tokens (attempt {retry_count + 1}/{self.answer_retries}). Retrying...")
        retry_count += 1
    logging.error(
        f"{self.name}'s {len(interaction_history) + 1} round answer for {pre_test_results[0]['question_id']} exceeded {max_allowed_tokens} tokens after {self.answer_retries} attempts. Returning last response.")
    # If after answer_retries attempts the response is still too long, return an empty string
    return response
