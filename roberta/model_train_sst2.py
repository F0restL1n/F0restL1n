from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch

# 加载本地数据集
data_files = {
    'train': './dataset/train.tsv',
    'dev': './dataset/dev.tsv',
    'test': './dataset/test.tsv'
}

dataset = load_dataset('csv', data_files=data_files, delimiter='\t')

# 加载 tokenizer 和模型
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 数据预处理
def preprocess_function(examples):
    return tokenizer(
        examples['sentence'], 
        padding='max_length',  # 填充到最大长度
        truncation=True,       # 截断超长文本
        max_length=512         # 设置最大长度（可以根据需要调整）
    )

# 预处理数据
train_dataset = dataset['train'].map(preprocess_function, batched=True)
val_dataset = dataset['dev'].map(preprocess_function, batched=True)
test_dataset = dataset['test'].map(preprocess_function, batched=True)

# 计算准确率
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    evaluation_strategy="epoch",     # 每个epoch后评估一次
    learning_rate=2e-5,              # 学习率
    per_device_train_batch_size=8,   # 每个设备的训练批次大小
    per_device_eval_batch_size=8,    # 每个设备的评估批次大小
    num_train_epochs=3,              # 训练轮数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=500,                # 每50步记录一次日志
    load_best_model_at_end=True,     # 训练结束时加载最好的模型
    metric_for_best_model="accuracy", # 使用准确度作为评估标准
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 训练
trainer.train()

# 评估模型
results = trainer.evaluate(test_dataset)
print(results)

# 保存模型
model.save_pretrained("./roberta-sst2")
tokenizer.save_pretrained("./roberta-sst2")

