from datasets import load_dataset

from config import DATASET_NAME, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def load_dataset_split():
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    return dataset


def preprocess_function(examples, tokenizer):
    inputs = ["question: " + str(item) for item in examples["input"]]
    targets = [str(item) for item in examples["output"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized_dataset
