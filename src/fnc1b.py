import numpy as np
import pandas as pd
import spacy
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from pandarallel import pandarallel
import multiprocessing
import os
import pyarrow.parquet as pq

# Download NLTK stopwords
nltk.download('stopwords')

# Paths
csv_path = "sampled_news.csv"
parquet_path = "sampled_news_sm.parquet"
output_parquet = "processed_fakenews.parquet"
output_csv = "processed_fakenews.csv"

# Convert CSV to Parquet if needed
if os.path.exists(parquet_path):
    data_path = parquet_path
elif os.path.exists(csv_path):
    print("🔄 Converting CSV to Parquet...")
    df = pd.read_csv(csv_path, lineterminator="\n", on_bad_lines="skip", usecols=["id", "content", "type"])
    df.to_parquet(parquet_path, index=False)
    print("✅ Conversion complete.")
    data_path = parquet_path
else:
    print("❌ Error: No dataset found.")
    exit()

# Load spaCy model
print("📚 Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import subprocess
    print("⬇️ Model not found. Downloading...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
print("📖 spaCy model loaded.")

# Stopwords & Stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Initialize parallel processing
pandarallel.initialize(nb_workers=max(1, int(multiprocessing.cpu_count() / 2)), progress_bar=True)

batch_size = 100000
parquet_file = pq.ParquetFile(data_path)

processed_chunks = []
vocab_before = Counter()
vocab_after_stopwords = Counter()
vocab_after_stemming = Counter()
total_words_before = 0
total_words_after_stopwords = 0
total_words_after_stemming = 0

total_chars_after_stopwords = 0
total_chars_after_stemming = 0

print("🧮 Processing text in batches...")
batch_num = 0
for batch in parquet_file.iter_batches(batch_size):
    print(f"🔢 Processing batch {batch_num + 1}...")
    chunk = batch.to_pandas()
    chunk = chunk.dropna(subset=["content"]).astype({'content': 'string'})

    print("🪙 Tokenizing text...")
    chunk_tokens = chunk["content"].parallel_apply(lambda text: [word.lower() for word in text.split() if word.isalpha()])
    for tokens in chunk_tokens:
        vocab_before.update(tokens)
        total_words_before += len(tokens)

    print("🚫 Removing stopwords...")
    chunk_no_stopwords = chunk_tokens.parallel_apply(lambda tokens: [word for word in tokens if word not in stop_words])
    for tokens in chunk_no_stopwords:
        vocab_after_stopwords.update(tokens)
        total_words_after_stopwords += len(tokens)
        total_chars_after_stopwords += sum(len(word) for word in tokens)

    print("🌱 Applying stemming...")
    chunk_stemmed = chunk_no_stopwords.parallel_apply(lambda tokens: [stemmer.stem(word) for word in tokens])
    for tokens in chunk_stemmed:
        vocab_after_stemming.update(tokens)
        total_words_after_stemming += len(tokens)
        total_chars_after_stemming += sum(len(word) for word in tokens)

    print("📝 Joining tokens back to text...")
    chunk["processed_text"] = chunk_stemmed.parallel_apply(lambda tokens: ' '.join(tokens))
    processed_chunks.append(chunk[["id", "processed_text", "type"]])
    batch_num += 1

# Save processed data
final_df = pd.concat(processed_chunks, ignore_index=True)
final_df.to_parquet(output_parquet, index=False)
final_df.to_csv(output_csv, index=False)

print(f"💾 Processed data saved to '{output_parquet}' and '{output_csv}'")

total_vocab_before = len(vocab_before)
total_vocab_after_stopwords = len(vocab_after_stopwords)
total_vocab_after_stemming = len(vocab_after_stemming)

total_stopword_reduction = (total_words_before - total_words_after_stopwords) / total_words_before * 100
print(f"📊 Total words (the raw number of all words in the text, including duplicates): {total_words_before:,}")
print(f"⏮️ Before stopword removal: {total_words_before:,}")
print(f"🔻 After stopword removal: {total_words_after_stopwords:,} (-{total_stopword_reduction:.2f}%)")

vocab_stemming_reduction = (total_vocab_after_stopwords - total_vocab_after_stemming) / total_vocab_after_stopwords * 100
print(f"🫆 Vocabulary (the number of distinct words in the text, ignoring duplicates):")
print(f"⏮️ Before stemming: {total_vocab_before:,}")
print(f"🔻 After stemming: {total_vocab_after_stemming:,} (-{vocab_stemming_reduction:.2f}%)")

avg_chars_after_stopwords = total_chars_after_stopwords / total_words_after_stopwords
avg_chars_after_stemming =  total_chars_after_stemming / total_words_after_stemming
avg_chars_reduction = (avg_chars_after_stopwords - avg_chars_after_stemming) / avg_chars_after_stopwords * 100
print(f"📏 Avg. length of retained words:")
print(f"⏮️ After stopword removal: {avg_chars_after_stopwords:.2f}")
print(f"🔻 After stemming: {avg_chars_after_stemming:.2f} (-{avg_chars_reduction:.2f}%)")

# Get most frequent words before and after stopword removal & stemming
def get_most_frequent_words(vocab, top_n=10):
    return vocab.most_common(top_n)

top_words_before = get_most_frequent_words(vocab_before)
top_words_after_stopwords = get_most_frequent_words(vocab_after_stopwords)
top_words_after_stemming = get_most_frequent_words(vocab_after_stemming)

print("📌 Top 10 words before preprocessing:", top_words_before)
print("📌 Top 10 words after stopword removal:", top_words_after_stopwords)
print("📌 Top 10 words after stemming:", top_words_after_stemming)

def plot_word_frequencies(vocab_before, vocab_after_stopwords, vocab_after_stemming, top_n=10000):
    plt.figure(figsize=(12, 7))
    
    freq_before = [freq for _, freq in vocab_before.most_common(top_n)]
    freq_after_stopwords = [freq for _, freq in vocab_after_stopwords.most_common(top_n)]
    freq_after_stemming = [freq for _, freq in vocab_after_stemming.most_common(top_n)]
    
    plt.loglog(range(1, len(freq_before)+1), freq_before, 
               label='Raw Text', color='royalblue', alpha=0.8, linewidth=2)
    plt.loglog(range(1, len(freq_after_stopwords)+1), freq_after_stopwords, 
               label='After Stopword Removal', color='orange', alpha=0.8, linewidth=2)
    plt.loglog(range(1, len(freq_after_stemming)+1), freq_after_stemming, 
               label='After Stemming', color='green', alpha=0.8, linewidth=2)
    
    # Add Zipf's law reference line
    zipf_x = np.array(range(1, top_n+1))
    zipf_y = freq_before[0] / zipf_x
    plt.plot(zipf_x, zipf_y, 'r--', label="Zipf's Law", alpha=0.5)
    
    top_words = [word for word, _ in vocab_before.most_common(5)]
    for rank, word in enumerate(top_words, 1):
        freq = vocab_before[word]
        plt.annotate(word, xy=(rank, freq), xytext=(rank*1.5, freq*1.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
    
    plt.title('Word Frequency Distribution (Log-Log Scale)', fontsize=14, pad=20)
    plt.xlabel('Word Rank (Log Scale)', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=11)
    
    plt.text(0.02, 0.02, 
             "• Steep drop at left = Stopwords dominate\n"
             "• Flatter curve after processing = Better balance\n"
             "• Close to Zipf's line = Natural language pattern", 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round", fc="white", ec="gray", pad=0.4))
    
    plt.tight_layout()
    plt.show()

plot_word_frequencies(vocab_before, vocab_after_stopwords, vocab_after_stemming)