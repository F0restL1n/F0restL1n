from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import torch

dataset = load_dataset("race", "middle")

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=4)

def preprocess_function(examples):
    questions = examples['question']
    options = examples['options']
    inputs = [f"{question} {option}" for question, option in zip(questions, options)]
    tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=512)
    labels = [ord(answer) - ord('A') for answer in examples['answer']]  
    tokenized_inputs['labels'] = labels 
    return tokenized_inputs

train_dataset = dataset["train"].map(preprocess_function, batched=True)
val_dataset = dataset["validation"].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir="./results",          
    evaluation_strategy="epoch",     
    learning_rate=2e-5,             
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,   
    num_train_epochs=3,             
    weight_decay=0.01,              
    logging_dir="./logs",        
    logging_steps=100,
    load_best_model_at_end=True     
)

def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,                        
    args=training_args,                  
    train_dataset=train_dataset,        
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics,    
    data_collator=data_collator        
)

trainer.train()

results = trainer.evaluate()
print("Evaluation results:", results)
