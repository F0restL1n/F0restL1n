import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

dataset = load_dataset("glue", "qqp")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def preprocess_function(examples):
    return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)
encoded_dataset = dataset.map(preprocess_function, batched=True)

#train_dataset = encoded_dataset["train"]
train_dataset = encoded_dataset['train'].shuffle(seed=42).select(range(len(dataset['train']) // 2))
eval_dataset = encoded_dataset["validation"]
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",          
    evaluation_strategy="epoch",     
    learning_rate=2e-5,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,  
    num_train_epochs=3,              
    weight_decay=0.01,               
    logging_dir="./logs",           
    logging_steps=5000,               
    load_best_model_at_end=True,   
)

def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=1)  
    return {"accuracy": accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,                     
    args=training_args,              
    train_dataset=train_dataset,     
    eval_dataset=eval_dataset,       
    compute_metrics=compute_metrics, 
)

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

