
from datasets import load_dataset
dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans")
print(dataset)
train_data = dataset['train']
train_data = train_data.select(range(5000))
print(train_data)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch
import datasets

# datasets.utils.logging.disable_progress_bar()

datasets.utils.logging.enable_progress_bar()
# Step 1: Load the CodeXGLUE Python-to-Java dataset
dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans")
train_data = dataset['train']
train_data = train_data.select(range(500))
# print(train_data)

# dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

# Step 2: Load a pre-trained model (CodeT5 or another seq2seq model)
model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    # Tokenize both input and target
    inputs = tokenizer(examples['java'], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples['cs'], padding="max_length", truncation=True, max_length=512)
    inputs['labels'] = targets['input_ids']
    return inputs

# Apply tokenization to the dataset
tokenized_dataset = train_data.map(tokenize_function, batched=True)
tokenized_dataset_test = dataset.map(tokenize_function, batched=True)
# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./codet5_java_to_cs",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
)

# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_test["test"],  # Optional, but good practice
    tokenizer=tokenizer,
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Save the model
model.save_pretrained("codet5_java_to_cs_model")
tokenizer.save_pretrained("codet5_java_to_cs_model")

# After fine-tuning, test the model for code translation
def generate_translation(input_code):
    inputs = tokenizer(input_code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(inputs['input_ids'].to(model.device), max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example: Translating Python to Java (you can replace this with SAS to Python after dataset creation)
translated_code = generate_translation("public class AddNumbers {public static int add(int a, int b) {return a + b;} public static void main(String[] args) {        int num1 = 5; int num2 = 10; int result = add(num1, num2); System.out.println(result);   }}")
print(f"Translated cs code: {translated_code}")