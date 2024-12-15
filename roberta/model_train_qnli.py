from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

dataset = load_dataset("glue", "qnli")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

def preprocess_function(examples):
    return tokenizer(examples['question'], examples['sentence'], padding='max_length', truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir="./results",            
    evaluation_strategy="epoch",       
    learning_rate=2e-5,               
    per_device_train_batch_size=16,    
    per_device_eval_batch_size=16,     
    num_train_epochs=3,               
    weight_decay=0.01,                
    logging_dir="./logs",             
    logging_steps=2000,                 
    load_best_model_at_end=True,       
    metric_for_best_model="accuracy", 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics, 
)

trainer.train()

eval_results = trainer.evaluate()

print(f"Evaluation results: {eval_results}")

trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
