import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import matplotlib.pyplot as plt

# 1. Load and preprocess LIAR dataset
print("📚 Loading LIAR dataset...")
liar_test = pd.read_csv("../data/liar_test_processed.csv")

# Binary label mapping (adjust based on your LIAR preprocessing)
print("🧮 Grouping into binary classes...")
liar_fake_labels = {'false', 'pants-fire'}  # Update with your actual LIAR labels
liar_test['label'] = liar_test.iloc[:, 1].apply(lambda x: 1 if x in liar_fake_labels else 0)

# 2. Load model and tokenizer
model_path = "./fake_news_bert"
print(f"⬇️ Loading model from {model_path}...")
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, do_lower_case=True)
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# 3. Tokenization
print("🪙 Tokenizing text...")
def tokenize_data(texts, max_length=512):
    results = {'input_ids': [], 'attention_mask': []}
    batch_size = 1000
    
    for i in tqdm(range(0, len(texts), batch_size, desc="Tokenizing")):
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

test_encodings = tokenize_data(liar_test['processed_text'].tolist())

# 4. Dataset Class
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

print("\n📝 Creating dataset...")
test_dataset = CustomDataset(test_encodings, liar_test['label'])

# 5. Prediction Function
def predict(model, dataset, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    preds = []
    true_labels = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
        # Get batch
        batch_indices = range(i, min(i+batch_size, len(dataset)))
        batch = [dataset[j] for j in batch_indices]
        
        # Prepare inputs
        inputs = {
            'input_ids': torch.stack([item['input_ids'] for item in batch]).to(device),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]).to(device)
        }
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Store results
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        true_labels.extend([item['labels'] for item in batch])
    
    return np.array(preds), np.array(true_labels)

# 6. Run Evaluation
print("\n🧪 Evaluating on LIAR test set...")
y_pred, y_true = predict(model, test_dataset)

# 7. Performance Report
print("\n📊 DistilBERT Performance on LIAR Dataset:")
print(classification_report(y_true, y_pred, target_names=['Reliable', 'Fake']))
