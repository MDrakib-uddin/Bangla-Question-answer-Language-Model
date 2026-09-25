import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from config import FINAL_MODEL_DIR


def load_pipeline():
    return pipeline(
        "text2text-generation",
        model=str(FINAL_MODEL_DIR),
        tokenizer=str(FINAL_MODEL_DIR),
    )


def ask_question(question: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(FINAL_MODEL_DIR)).to(device)

    prompt = "question: " + question
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    sample_question = "বাংলাদেশের রাজধানী কী?"
    print("Question:", sample_question)
    print("Answer:", ask_question(sample_question))
