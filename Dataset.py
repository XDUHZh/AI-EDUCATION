class MMLU_PRO(BaseDataset):
    @classmethod
    def load_data(cls, dataset_name: str = "TIGER-Lab/MMLU-Pro"):
        dataset = load_dataset(dataset_name)
        test_df, val_df = dataset["test"], dataset["validation"]
        test_df = cls.preprocess_data(test_df)
        val_df = cls.preprocess_data(val_df)
        return test_df, val_df

    @classmethod
    def preprocess_data(cls, data):
        categorized_data = defaultdict(list)
        for entry in data:
            categorized_data[entry["category"]].append(
                {
                    "question_id": str(entry["question_id"]),
                    "question": entry["question"],
                    "options": [opt for opt in entry["options"] if opt != "N/A"],
                    "answer": entry["answer"],
                    "answer_index": entry["answer_index"],
                    "cot_content": entry["cot_content"],
                    "category": entry["category"],
                }
            )
        return dict(categorized_data)


class GPQA(BaseDataset):
    SEED = 42

    @classmethod
    def load_data(cls, test_data_folder_path: str, val_data_filepath: str, dataset_name: str = "gpqa_diamond.csv"):
        random.seed(cls.SEED)

        test_df = pd.read_csv(os.path.join(test_data_folder_path, dataset_name))
        test_df = cls.preprocess_data(test_df)

        with open(val_data_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        val_df = defaultdict(list)

        for question in data["questions"]:
            options = list(question["choices"].values())
            correct_answer = question["choices"][question["correct_answer"]]
            val_df["general"].append(
                {
                    "question_id": str(hash(question["question"])),
                    "question": question["question"],
                    "options": options,
                    "answer": correct_answer,
                    "answer_index": options.index(correct_answer),
                    "cot_content": question["explanation"],
                    "category": "general",
                }
            )

        val_df = dict(val_df)

        return test_df, val_df

    @classmethod
    def preprocess_data(cls, data: pd.DataFrame):
        INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
        categorized_data = defaultdict(list)
        for _, row in data.iterrows():
            options = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"]
            ]
            random.shuffle(options)
            correct_index = options.index(row["Correct Answer"])

            categorized_data[row["High-level domain"]].append(
                {
                    "question_id": str(row["Record ID"]),
                    "question": row["Question"],
                    "options": options,
                    "answer": INDEX_TO_LETTER[correct_index],
                    "answer_index": correct_index,
                    "cot_content": row["Explanation"],
                    "category": row["High-level domain"],
                }
            )
        return dict(categorized_data)
