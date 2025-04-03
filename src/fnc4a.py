import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

input_path = "../data/liar"
output_path = "../data/liar_processed"

# Initialize preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Apply the same process as FakeNewsCorpus
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Tokenization
    tokens = [word.lower() for word in text.split() if word.isalpha()]
    
    # Stopword removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Load LIAR dataset
print("🔍 Loading LIAR dataset...")
liar_test = pd.read_csv(input_path + "_test.tsv", sep='\t', header=None)

# Apply preprocessing (column 2 contains the text statements)
print("🪙 Preprocessing LIAR text...")
liar_test['processed_text'] = liar_test[2].apply(preprocess_text)

# Save preprocessed data
liar_test.to_csv(output_path + "_test.tsv", index=False)
print(f"💾 Preprocessed LIAR data saved to '{output_path + "_test.tsv"}'")