# Import required libraries with optimized imports
import multiprocessing
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report
import numpy as np
import os
from pandarallel import pandarallel

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True if torch.cuda.is_available() else False
torch.backends.cudnn.allow_tf32 = True if torch.cuda.is_available() else False

# Optimized data loading
def load_split(file_prefix, split_name):
    parquet_path = f"{file_prefix}_{split_name}.parquet"
    csv_path = f"{file_prefix}_{split_name}.csv"
    
    if os.path.exists(parquet_path):
        print(f"📚 Loading data from Parquet file at '{parquet_path}'")
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"🔄 Loading from CSV at '{csv_path}'")
        return pd.read_csv(csv_path)
    else:
        print(f"❌ Error: Neither Parquet nor CSV file found at {parquet_path} or {csv_path}")
        exit()

# Load data in parallel
print("🔄 Loading datasets...")
train = load_split("../data/sampled_fakenews", "train")
val = load_split("../data/sampled_fakenews", "valid") 
test = load_split("../data/sampled_fakenews", "test")

# Precompute fake labels set for faster lookup
print("\n🧮 Grouping into binary classes...")
FAKE_LABELS = {'fake', 'conspiracy', 'rumor', 'unreliable', 'junksci', 'hate', 'satire', 'clickbait'}
for df in [train, val, test]:
    df['label'] = df['type'].isin(FAKE_LABELS).astype(int)

# Check dataset for duplicates
def verify_data(dataset):
    print("\n🔍 Verifying dataset...")
    print(f"Labels distribution:\n{dataset['label'].value_counts()}")
    print(f"Sample texts:\n{dataset['processed_text'].head(3)}")

    print(dataset['label'].value_counts().sort_index().values)
    
    # Count duplicates
    dup_counts = dataset['processed_text'].value_counts()
    dup_counts = dup_counts[dup_counts > 1]  # Only keep duplicates
    
    print(f"\n📊 Duplicate text statistics:")
    print(f"Total duplicate texts: {len(dup_counts)}")
    print(f"Total duplicate occurrences: {dup_counts.sum() - len(dup_counts)}")
    
    if not dup_counts.empty:
        print("\n🔢 Top 5 most frequent duplicates:")
        for text, count in dup_counts.head(5).items():
            print(f"{count} x {text[:100]}... ")
        
        # Show conflicting labels (same text, different labels)
        conflicts = dataset.groupby('processed_text')['label'].nunique()
        conflicts = conflicts[conflicts > 1]
        
        if not conflicts.empty:
            print("\n⚠️ Label conflicts found (same text, different labels):")
            for text in conflicts.head(3).index:
                labels = dataset[dataset['processed_text'] == text]['label'].unique()
                print(f"  '{text[:50]}...' has labels: {labels}")
        
        # Remove duplicates (keep first occurrence)
        print(f"\n🛠️ Removing {len(dup_counts)} duplicates...")
        dataset.drop_duplicates(subset=['processed_text'], keep='first', inplace=True)
    
    return dataset

verify_data(train)
verify_data(val)
verify_data(test)

# Initialize tokenizer
print("\n🪙 Tokenizing text (this may take a while)...")
pandarallel.initialize(nb_workers=max(1, int(multiprocessing.cpu_count())), progress_bar=True)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)

def tokenize_data(texts, max_length=512):
    results = {'input_ids': [], 'attention_mask': []}
    batch_size = 1000
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", unit="batch"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        results['input_ids'].append(encoded['input_ids'])
        results['attention_mask'].append(encoded['attention_mask'])
    
    return {
        'input_ids': torch.cat(results['input_ids']),
        'attention_mask': torch.cat(results['attention_mask'])
    }

train_encodings = tokenize_data(train['processed_text'].tolist())
val_encodings = tokenize_data(val['processed_text'].tolist())
test_encodings = tokenize_data(test['processed_text'].tolist())

# Create dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)

print("\n📝 Creating datasets...")
train_dataset = CustomDataset(train_encodings, train['label'])
val_dataset = CustomDataset(val_encodings, val['label'])
test_dataset = CustomDataset(test_encodings, test['label'])

# Load pretrained  model
print("\n⬇️ Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torch_dtype=torch.float32
)

# Initialize model weights according label distribution
class_counts = torch.tensor(train['label'].value_counts().sort_index().values)
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum()
model.loss_fct = nn.CrossEntropyLoss(weight=class_weights.to('cuda' if torch.cuda.is_available() else 'cpu'))

with torch.no_grad():
    for layer in [model.pre_classifier, model.classifier]:
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(layer.bias)

# Check model parameters
print("\n🧠 Model parameter check:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.norm().item():.4f} (mean={param.data.mean().item():.4f})")

# Enable mixed precision training if GPU available
if torch.cuda.is_available():
    model = model.to('cuda')
    print("\n🚀 Using GPU acceleration with mixed precision")
    scaler = torch.amp.GradScaler('cuda')

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Precision
    # !WARNING: Check GPU compatibility for each of these options
    fp16=False,
    bf16=True,
    tf32=True,
    
    # Scheduling
    lr_scheduler_type="linear",
    optim="adamw_torch",
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    # System
    gradient_checkpointing=False,
    dataloader_num_workers=0,
    report_to="none",
    seed=42
)

# Simplified metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = (preds[labels == 1] == 1).mean()
    recall = (labels[preds == 1] == 1).mean()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Add early stopping callback
trainer.add_callback(EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.02,
))

# Verify tokenization
sample = train_dataset[0]
print("\n💬 Sample input IDs:", sample['input_ids'][:10])
print("💬 Sample attention mask:", sample['attention_mask'][:10])
print("💬 Sample label:", sample['labels'])

# Verify model can process a sample
with torch.no_grad():
    output = model(
        input_ids=sample['input_ids'].unsqueeze(0).to('cuda'),
        attention_mask=sample['attention_mask'].unsqueeze(0).to('cuda')
    )
    print("\n✅ Model output check:", output.logits)

# Train with progress bar
print("\n🏋️ Training the model...")
trainer.train(resume_from_checkpoint=False) # Set to True to resume training from the last checkpoint saved in ./results, in case of interruptions. Default is False

# Optimized evaluation
print("\n🧪 Evaluating on validation set...")
print(trainer.evaluate(val_dataset))

print("\n🧪 Evaluating on test set...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test['label'].values

print("\n📊 Final Test Performance:")
print(classification_report(y_true, y_pred, target_names=['Reliable', 'Fake']))

# Save the model efficiently
print("\n💾 Saving model...")
model.save_pretrained("./fake_news_bert", safe_serialization=True)
tokenizer.save_pretrained("./fake_news_bert")