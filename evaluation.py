import datasets
import transformers
import evaluate

# turn on logging to see the progress
evaluate.logging.set_verbosity_debug()
datasets.logging.set_verbosity_debug()
transformers.logging.set_verbosity_debug()

model_name = "mrm8488/longformer-base-4096-finetuned-squadv2"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
pipeline = transformers.pipeline("question-answering", model=model, tokenizer=tokenizer)

data = datasets.load_dataset('LongPolicyQA', split='validation', cache_dir='./.cache')
subset = data.select(range(1))
squad_metric = evaluate.load("squad")
task_evaluator = evaluate.evaluator("question-answering")


eval_results = task_evaluator.compute(
    model_or_pipeline=pipeline,
    data=subset,
    squad_v2_format=True,
    metric="squad_v2"
)
print(eval_results)