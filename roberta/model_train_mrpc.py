from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# 1. 加载 MRPC 数据集
dataset = load_dataset("glue", "mrpc")

# 2. 加载 RoBERTa 模型和分词器
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 3. 数据预处理函数：将句子对 (sentence1, sentence2) 转换为模型输入格式
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', truncation=True, max_length=128)

# 对数据集进行映射和预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 4. 定义评估函数
def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(axis=1)
    
    # 计算准确率、F1分数和Matthews相关系数
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    matthews = matthews_corrcoef(labels, preds)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "matthews_corrcoef": matthews
    }

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",            # 输出目录
    evaluation_strategy="epoch",       # 每个 epoch 进行评估
    learning_rate=2e-5,                # 学习率
    per_device_train_batch_size=16,    # 训练 batch size
    per_device_eval_batch_size=16,     # 评估 batch size
    num_train_epochs=3,                # 训练 epoch 数
    weight_decay=0.01,                 # 权重衰减
    logging_dir="./logs",              # 日志目录
    logging_steps=1000,                  # 每多少步记录一次
    load_best_model_at_end=True,       # 训练结束后加载最佳模型
    metric_for_best_model="accuracy", # 按准确率选择最佳模型
)

# 6. 使用 Trainer API 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,   # 评估函数
)

# 7. 开始训练
trainer.train()

# 8. 评估模型
eval_results = trainer.evaluate()

# 9. 打印评估结果
print(f"Evaluation results: {eval_results}")

# 10. 保存模型和分词器
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
