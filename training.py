import datasets
import transformers
import evaluate
import torch
import wandb

wandb.init(project="longformer")

# turn on logging to see the progress
evaluate.logging.set_verbosity_debug()
datasets.logging.set_verbosity_debug()
transformers.logging.set_verbosity_debug()

model_name = "mrm8488/longformer-base-4096-finetuned-squadv2"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
pipeline = transformers.pipeline("question-answering", model=model, tokenizer=tokenizer)

data = datasets.load_dataset('LongPolicyQA', cache_dir='./.cache')

tokenized_dataset = data.map(
    lambda examples: tokenizer(examples["context"], examples["question"], truncation=True),
    batched=True,
    desc="Running tokenizer on dataset",
)

training_args = transformers.TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    report_to="wandb",
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics = lambda p: evaluate.load("squad_v2").compute(predictions=p, references=tokenized_dataset["test"])
)


# print out device information

print("Device:", torch.cuda.current_device())
print("Available devices:", torch.cuda.device_count())

trainer.train()