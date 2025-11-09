class TeacherLLM(BaseLLM):
    def __init__(
            self,
            name: str,
            model: str,
            api_key: str,
            base_url: str,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            use_few_shot: bool = True,
            num_if_few_shots: int = 5,
            recommended_question_token_limit: int = 150,
            recommended_education_theory: Optional[str] = None,
            max_tokens_rerun_threshold_percentage: float = 0.8,
            question_retries: int = 3,
            is_vertex_ai: bool = False,
            project_id: str = None,
            location: str = None,
    ):
        self.is_vertex_ai = is_vertex_ai
        self.project_id = project_id
        self.location = location
        self.token_expiry = 0

        if self.is_vertex_ai:
            self.refresh_token()
            base_url = self.client.base_url
            api_key = self.client.api_key

        super().__init__(name, model, api_key, base_url, temperature, max_tokens, use_few_shot, num_if_few_shots)

        self.recommended_question_token_limit = recommended_question_token_limit
        self.recommended_education_theory = recommended_education_theory
        self.max_tokens_rerun_threshold_percentage = max_tokens_rerun_threshold_percentage
        self.question_retries = question_retries

    def refresh_token(self):
        credentials, _ = google.auth.default()
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        self.client = OpenAI(
            base_url=f'https://{self.location}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi',
            api_key=credentials.token
        )
        self.token_expiry = time.time() + 3540  # Set expiry to 59 minutes from now

    def generate_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if self.is_vertex_ai and time.time() > self.token_expiry:
            self.refresh_token()
        return super().generate_response(messages)

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            # "api_key": self.api_key,
            # "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_few_shot": self.use_few_shot,
            "num_if_few_shots": self.num_if_few_shots,
            "recommended_question_token_limit": self.recommended_question_token_limit,
            "recommended_education_theory": self.recommended_education_theory,
            "max_tokens_rerun_threshold_percentage": self.max_tokens_rerun_threshold_percentage,
            "question_retries": self.question_retries,
            "is_vertex_ai": self.is_vertex_ai,
            "project_id": self.project_id,
            "location": self.location,
        }

    def generate_question(
            self,
            category: str,
            pre_test_results: List[Dict[str, Any]],
            interaction_history: List[Dict[str, str]],
            current_round: int,
            total_rounds: int,
            few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        system_message = f"""You are an expert teacher in {category} {"using the " + self.recommended_education_theory + " approach " if self.recommended_education_theory else ""}dedicated to enhancing the student's understanding after analyzing the student's response to a pre-test. 
        Your task is to ask {total_rounds} rounds of relevant, thought-provoking questions to the student. 
        You should ask one new question per round (and if needed, provide necessary corrections or feedback for the student's previous round's answers), 
        each under {self.recommended_question_token_limit} tokens, without revealing the correct answers or specific details of the pre-test questions. 
        Your goal is to prepare the student for the post-test by fostering a deeper and more comprehensive understanding of the subject matter.\n\n"""

        few_shot_examples_message = ""
        if self.use_few_shot and few_shot_examples:
            few_shot_examples_message = "\n\nHere are some example questions and reasoning processes:\n"
            for example in few_shot_examples[:self.num_if_few_shots]:
                few_shot_examples_message += f"Question: {example['question']}\nReasoning: {example['cot_content']}\n\n"

        pre_test_info = "\n\nHere are the pre-test results of the student:\n"
        for r in pre_test_results:
            pre_test_info += f"""
                Question ID: {r['question_id']}
                Question: {r['question']}
                Student's Reasoning: {r['model_response']}
                Student's Answer: {r['model_prediction']}
                Student's Answer is Correct or Not: {"Correct." if r['correct_answer'] == r['model_prediction'] else "Incorrect."}

                """

        messages = [
            {"role": "system", "content": system_message + "\n" + few_shot_examples_message + "\n" + pre_test_info}]

        for interaction in interaction_history:
            messages.append(
                {"role": "assistant", "content": f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', interaction['question'])}"}
            )
            messages.append(
                {"role": "user", "content": f"Student: {re.sub(r'^(Student:( )*)+', '', interaction['answer'])}"}
            )

        messages.append(
            {"role": "user",
             "content": f"Generate the round {current_round} question ({self.recommended_question_token_limit} tokens or less) to promote better understanding:", }
        )

        # Initialize retry counter, when the output exceeds 80% of max_tokens and is greater than recommended_question_token_limit, a rerun of generation is required.

        retry_count = 0

        while retry_count < self.question_retries:
            response = self.generate_response(messages)
            response_tokens = self.count_tokens(response)
            max_allowed_tokens = max(
                self.max_tokens * self.max_tokens_rerun_threshold_percentage,
                self.recommended_question_token_limit,
            )

            if 0 < response_tokens <= max_allowed_tokens:
                return response
            elif response_tokens == 0:
                logging.warning(
                    f"{self.name}'s {current_round}/{total_rounds} round question for {pre_test_results[0]['question_id']} had 0 tokens (attempt {retry_count + 1}/{self.question_retries}). Retrying...")
            else:
                logging.warning(
                    f"{self.name}'s {current_round}/{total_rounds} round question for {pre_test_results[0]['question_id']} had {response_tokens} tokens and exceeded {max_allowed_tokens} tokens (attempt {retry_count + 1}/{self.question_retries}). Retrying...")
            retry_count += 1

        logging.error(
            f"{self.name}'s {current_round}/{total_rounds} round question for {pre_test_results[0]['question_id']} exceeded {max_allowed_tokens} tokens after {self.question_retries} attempts. Returning last response.")
        # If after question_retries attempts the response is still too long, return an empty string
        return response