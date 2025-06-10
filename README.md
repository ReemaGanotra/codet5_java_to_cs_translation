# codet5_java_to_cs_translation
Fine-tuning the CodeT5 model for code-to-code translation using the CodeXGLUE dataset (https://huggingface.co/datasets/google/code_x_glue_cc_code_to_code_trans). This project demonstrates translating Java code into C# (cs) by leveraging Hugging Face Transformers, datasets, and the Salesforce/codet5-base model.

## 🔧 Features

- Loads and preprocesses the CodeXGLUE Java-to-C# dataset
- Fine-tunes a pre-trained sequence-to-sequence model (CodeT5)
- Evaluates performance on a test split
- Generates translated code samples

## 📦 Dependencies

- `transformers`
- `datasets`
- `torch`

Install dependencies via:

```bash
pip install transformers datasets torch
