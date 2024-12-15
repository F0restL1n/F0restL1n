from transformers import RobertaTokenizer, RobertaForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch

dataset = load_dataset("squad")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForQuestionAnswering.from_pretrained("roberta-base")

def preprocess_function(examples):
    inputs = tokenizer(examples['question'], examples['context'], truncation=True, padding="max_length", max_length=512)

    start_positions = []
    end_positions = []
    
    for answer in examples['answers']:
        answer_text = answer['text'][0]  
        answer_start = answer['answer_start'][0]  
        answer_end = answer_start + len(answer_text) - 1  
        encoding = tokenizer.encode_plus(examples['question'], examples['context'], truncation=True, padding="max_length", max_length=512)
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        token_start = None
        token_end = None
        
        for idx in range(len(encoding['input_ids']) - len(answer_tokens) + 1):
            if encoding['input_ids'][idx:idx+len(answer_tokens)] == answer_tokens:
                token_start = idx
                token_end = idx + len(answer_tokens) - 1
                break
        
        if token_start is None or token_end is None:
            token_start = token_end = tokenizer.model_max_length - 1 
        
        start_positions.append(token_start)
        end_positions.append(token_end)
    
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

#train_dataset = dataset["train"].map(preprocess_function, batched=True)
train_dataset = dataset['train'].map(preprocess_function, batched=True).shuffle(seed=42).select(range(len(dataset['train']) // 2))
val_dataset = dataset["validation"].map(preprocess_function, batched=True)

def compute_metrics(p):
    predictions, labels = p
    start_preds = torch.argmax(torch.tensor(predictions[0]), dim=-1)
    end_preds = torch.argmax(torch.tensor(predictions[1]), dim=-1)
    start_labels = torch.tensor(labels[0]) 
    end_labels = torch.tensor(labels[1])
    start_accuracy = (start_preds == start_labels).float().mean().item()
    end_accuracy = (end_preds == end_labels).float().mean().item()
    start_f1 = f1_score(start_labels.numpy(), start_preds.numpy(), average='macro')
    end_f1 = f1_score(end_labels.numpy(), end_preds.numpy(), average='macro')
    eval_f1 = (start_f1 + end_f1) / 2 
    return {
        "eval_f1": eval_f1 
    }

training_args = TrainingArguments(
    output_dir="./results",          
    evaluation_strategy="epoch",    
    save_strategy="epoch", 
    learning_rate=3e-5,             
    per_device_train_batch_size=16,   
    per_device_eval_batch_size=16,    
    num_train_epochs=3,             
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=5000,                
    load_best_model_at_end=True,    
    metric_for_best_model="eval_f1",   
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

model.save_pretrained("./roberta-squad")
tokenizer.save_pretrained("./roberta-squad")
