from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score

dataset = load_dataset("glue", "mnli")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128)

train_dataset = dataset["train"].map(preprocess_function, batched=True)
val_dataset = dataset["validation_matched"].map(preprocess_function, batched=True)

def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./results",          
    evaluation_strategy="epoch",    
    save_strategy="epoch",
    learning_rate=2e-5,           
    per_device_train_batch_size=16,   
    per_device_eval_batch_size=16,    
    num_train_epochs=3,             
    weight_decay=0.01,              
    logging_dir='./logs',           
    logging_steps=1000,                
    load_best_model_at_end=True,     
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)

model.save_pretrained("./roberta-mnli")
tokenizer.save_pretrained("./roberta-mnli")
