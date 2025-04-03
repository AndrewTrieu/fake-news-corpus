import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

input_path = "../data/sampled_fakenews"

# Function to perform hyperparameter tuning, not used in the final script
def hyperparameter_tuning():
    print("🔍 Hyperparameter tuning...")
    param_grid_lr = {
        'C': [0.1, 1, 10],
        'max_iter': [100, 500, 1000],
        'class_weight': ['balanced', None]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid_lr, cv=3)
    grid.fit(X_train, y_train)
    print("✅ Best Logistic Regression Parameters:", grid.best_params_)

    param_grid_nb = {
        'alpha': [0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
    }
    grid_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=3) 
    grid_nb.fit(X_val, y_val)
    print("✅ Best Naïve Bayes Parameters:", grid_nb.best_params_)

#---FAKENEWSCORPUS DATASET---

# Load parquet first, fall back to CSV if not available
def load_split(file_prefix, split_name):
    try:
        print(f"📚 Loading data from Parquet file at '{file_prefix}_{split_name}.parquet'")
        return pd.read_parquet(f"{file_prefix}_{split_name}.parquet")
    except FileNotFoundError:
        print(f"❌ Error: Neither Parquet nor CSV file found at {file_prefix}_{split_name}.parquet or {file_prefix}_{split_name}.csv")
        exit()
    except:
        print(f"🔄 Parquet file not found, loading from CSV at '{file_prefix}_{split_name}.csv'")
        return pd.read_csv(f"{file_prefix}_{split_name}.csv")

train = load_split(input_path, "train")
val = load_split(input_path, "valid") 
test = load_split(input_path, "test")

# "Political" and "bias" may not be inherently fake, and "unknown" is neutral
print("🧮 Grouping into binary classes...")
fake_labels = {'fake', 'conspiracy', 'rumor', 'unreliable', 'junksci', 'hate', 'satire', 'clickbait'}
for df in [train, val, test]:
    df['label'] = df['type'].apply(lambda x: 1 if x in fake_labels else 0)

print("🪙 Preprocessing text...")
tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(train['processed_text'])
X_val = tfidf.transform(val['processed_text'])
X_test = tfidf.transform(test['processed_text'])

y_train = train['label']
y_val = val['label']
y_test = test['label']

print("🔍 Training models...")
lr = LogisticRegression(C=10, max_iter=100, class_weight=None, random_state=42)
lr.fit(X_train, y_train)

nb = MultinomialNB(alpha=0.1, fit_prior=True)
nb.fit(X_train, y_train)

y_test_pred_lr = lr.predict(X_test)
print("\n📊 Logistic Regression FakeNewsCorpus Performance:")
print(classification_report(y_test, y_test_pred_lr, target_names=['Reliable', 'Fake']))

y_test_pred_nb = nb.predict(X_test)
print("\n📊 Naïve Bayes FakeNewsCorpus Performance:")
print(classification_report(y_test, y_test_pred_nb, target_names=['Reliable', 'Fake']))

#---LIAR DATASET---
# Load the tsv file
print("📚 Loading LIAR dataset...")
liar_test = pd.read_csv("../data/liar_test_processed.csv")

# "Political" and "bias" may not be inherently fake, and "unknown" is neutral
print("🧮 Grouping into binary classes...")
liar_fake_labels = {'false', 'pants-fire'}
liar_test['label'] = liar_test.iloc[:, 1].apply(lambda x: 1 if x in liar_fake_labels else 0)

# Check for NaN values in processed_text
liar_test = liar_test.dropna(subset=['processed_text'])

# Transform LIAR text using the same TF-IDF vectorizer
print("🪙 Preprocessing text...")
X_liar_test = tfidf.transform(liar_test['processed_text'])

# Logistic Regression
y_liar_pred_lr = lr.predict(X_liar_test)
print("\n📊 Logistic Regression LIAR Performance:")
print(classification_report(liar_test['label'], y_liar_pred_lr, target_names=['Reliable', 'Fake']))

# Naïve Bayes
y_liar_pred_nb = nb.predict(X_liar_test)
print("\n📊 Naïve Bayes LIAR Performance:")
print(classification_report(liar_test['label'], y_liar_pred_nb, target_names=['Reliable', 'Fake']))

