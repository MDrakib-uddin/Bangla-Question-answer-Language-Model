import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from config import (
    FINAL_MODEL_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_INPUT_LENGTH,
    MODEL_NAME,
    NUM_TRAIN_EPOCHS,
    OUTPUT_DIR,
    TRAIN_BATCH_SIZE,
)
from data import load_dataset_split, tokenize_dataset


def build_training_args():
    return TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )


def train_model():
    dataset = load_dataset_split()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = build_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    print(f"Training completed. Model saved to: {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    train_model()
