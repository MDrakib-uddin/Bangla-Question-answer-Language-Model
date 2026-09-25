# Bangla Question Answer Language Model

This project is a lightweight Bangla question-answering model training pipeline. It fine-tunes the `google/mt5-small` model on a Bangla QA dataset so it can generate answers for user questions in Bengali.

## Features
- Uses a Bangla question-answer dataset
- Built on a `mt5-small` sequence-to-sequence model
- Keeps data processing, training, inference, config, and Hugging Face upload logic in separate Python files
- Easy to run for training, prediction, and model publishing

## Project Structure

```bash
bangla-question-answer-language-model/
├── config.py
├── data.py
├── train.py
├── predict.py
├── upload_to_hub.py
├── requirements.txt
├── README.md
├── question-answer.ipynb
├── LICENSE
└── qa-mt5-final/                  # trained model output
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

This script loads the dataset, preprocesses it, trains the model, and saves the final model in the `qa-mt5-final` folder.

## Generate Answers

```bash
python predict.py
```

You can change the `sample_question` variable in `predict.py` to test different questions.

## Upload to Hugging Face

```bash
python upload_to_hub.py
```

This script uploads the trained model to a Hugging Face repository using `huggingface_hub`. You must sign in beforehand using `huggingface-cli login` or the `login()` function.

## Dataset

This project uses the following dataset:

- `rasheduzzaman/Bangla_question_answer_pair_70K_dataset`

## Model

- `google/mt5-small`

## Reference Note

The `question-answer.ipynb` notebook contains the original experiment and training workflow used during development. It includes data loading, preprocessing, training, inference, and publishing steps.

## Future Improvements
- Hyperparameter tuning to improve model accuracy
- Use a larger Bangla QA dataset
- Add a Web UI or Streamlit app
- Improve answer generation quality

## License

For license details, please see [LICENSE](LICENSE).