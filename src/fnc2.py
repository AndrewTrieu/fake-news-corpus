import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

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

train = load_split("../data/sampled_fakenews", "train")
val = load_split("../data/sampled_fakenews", "valid") 
test = load_split("../data/sampled_fakenews", "test")

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
print("\n📊 Logistic Regression Test Performance:")
print(classification_report(y_test, y_test_pred_lr, target_names=['Reliable', 'Fake']))

y_test_pred_nb = nb.predict(X_test)
print("\n📊 Naïve Bayes Test Performance:")
print(classification_report(y_test, y_test_pred_nb, target_names=['Reliable', 'Fake']))

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
