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
import pyarrow as pa
from datetime import datetime

# Print log messages with timestamp
def print_log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Download NLTK stopwords
nltk.download('stopwords')

# Load spaCy model
print_log("📚 Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import subprocess
    print_log("⬇️ Model not found. Downloading...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
print_log("📖 spaCy model loaded.")

# Paths
input_path = "../data/news_cleaned_2018_02_13"
output_path = "../data/processed_fakenews"

# Convert CSV to Parquet if needed
if os.path.exists(input_path + ".parquet"):
    data_path = input_path + ".parquet" 
elif os.path.exists(input_path + ".csv"):
    print_log("🔄 Converting CSV to Parquet...")
    
    chunksize=1e5
    pqwriter = None
    for i, df in enumerate(pd.read_csv(input_path + ".csv", lineterminator="\n", on_bad_lines="skip", chunksize=chunksize, usecols=["id", "content", "type"])):
        table = pa.Table.from_pandas(df)
        # If it's the first chunk, create a new parquet writer
        if i == 0:
            pqwriter = pq.ParquetWriter(input_path + ".parquet", table.schema)            
        pqwriter.write_table(table)

    if pqwriter:
        pqwriter.close()

    print_log("✅ Conversion complete.")
    data_path = input_path + ".parquet"
else:
    print_log("❌ Error: No dataset found.")
    exit()

# Stopwords & Stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Initialize parallel processing
# !WARNING: This will use all available CPU cores, might kill host machine
# Set progress_bar=True to see a progress bar
pandarallel.initialize(nb_workers=max(1, int(multiprocessing.cpu_count())), progress_bar=False)

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

# Process text in batches
print_log("🧮 Processing text in batches...")
batch_num = 0
for batch in parquet_file.iter_batches(batch_size):
    print_log(f"🔢 Processing batch {batch_num + 1}...")
    chunk = batch.to_pandas()
    chunk = chunk.dropna(subset=["content"]).astype({'content': 'string'})

    # Tokenize, remove stopwords, and apply stemming
    print_log("🪙 Tokenizing text...")
    chunk_tokens = chunk["content"].parallel_apply(lambda text: [word.lower() for word in text.split() if word.isalpha()])
    for tokens in chunk_tokens:
        vocab_before.update(tokens)
        total_words_before += len(tokens)

    print_log("🚫 Removing stopwords...")
    chunk_no_stopwords = chunk_tokens.parallel_apply(lambda tokens: [word for word in tokens if word not in stop_words])
    for tokens in chunk_no_stopwords:
        vocab_after_stopwords.update(tokens)
        total_words_after_stopwords += len(tokens)
        total_chars_after_stopwords += sum(len(word) for word in tokens)

    print_log("🌱 Applying stemming...")
    chunk_stemmed = chunk_no_stopwords.parallel_apply(lambda tokens: [stemmer.stem(word) for word in tokens])
    for tokens in chunk_stemmed:
        vocab_after_stemming.update(tokens)
        total_words_after_stemming += len(tokens)
        total_chars_after_stemming += sum(len(word) for word in tokens)

    # Join tokens back to text
    print_log("📝 Joining tokens back to text...")
    chunk["processed_text"] = chunk_stemmed.parallel_apply(lambda tokens: ' '.join(tokens))
    processed_chunks.append(chunk[["id", "processed_text", "type"]])
    batch_num += 1

# Save processed data
final_df = pd.concat(processed_chunks, ignore_index=True)
final_df.to_parquet(output_path + ".parquet", index=False)
final_df.to_csv(output_path + ".csv", index=False)

print_log(f"💾 Processed data saved to '{output_path + ".parquet"}' and '{output_path + ".csv"}'")

# Print statisticsoutput_csv, index=False)
total_vocab_before = len(vocab_before)
total_vocab_after_stopwords = len(vocab_after_stopwords)
total_vocab_after_stemming = len(vocab_after_stemming)

total_stopword_reduction = (total_words_before - total_words_after_stopwords) / total_words_before * 100
print_log(f"📊 Total words (the raw number of all words in the text, including duplicates): {total_words_before:,}")
print(f"⏮️ Before stopword removal: {total_words_before:,}")
print(f"🔻 After stopword removal: {total_words_after_stopwords:,} (-{total_stopword_reduction:.2f}%)")

vocab_stemming_reduction = (total_vocab_after_stopwords - total_vocab_after_stemming) / total_vocab_after_stopwords * 100
print_log(f"🫆 Vocabulary (the number of distinct words in the text, ignoring duplicates):")
print(f"⏮️ Before stemming: {total_vocab_before:,}")
print(f"🔻 After stemming: {total_vocab_after_stemming:,} (-{vocab_stemming_reduction:.2f}%)")

avg_chars_after_stopwords = total_chars_after_stopwords / total_words_after_stopwords
avg_chars_after_stemming =  total_chars_after_stemming / total_words_after_stemming
avg_chars_reduction = (avg_chars_after_stopwords - avg_chars_after_stemming) / avg_chars_after_stopwords * 100
print_log(f"📏 Avg. length of retained words:")
print(f"⏮️ After stopword removal: {avg_chars_after_stopwords:.2f}")
print(f"🔻 After stemming: {avg_chars_after_stemming:.2f} (-{avg_chars_reduction:.2f}%)")

# Get most frequent words before and after stopword removal & stemming
def get_most_frequent_words(vocab, top_n=10):
    return vocab.most_common(top_n)

top_words_before = get_most_frequent_words(vocab_before)
top_words_after_stopwords = get_most_frequent_words(vocab_after_stopwords)
top_words_after_stemming = get_most_frequent_words(vocab_after_stemming)

print_log("📌 Top 10 words:")
print("🔝 Before preprocessing:", top_words_before)
print("🔝 After stopword removal:", top_words_after_stopwords)
print("🔝 After stemming:", top_words_after_stemming)

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