import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings("ignore")

INPUT_CSV  = "/Users/Ishita/Desktop/IDS_570_final/trade_occurrences.csv"
OUTPUT_CSV = "/Users/Ishita/Desktop/IDS_570_final/trade_bert_clusters.csv"
OUTPUT_PLOT = "/Users/Ishita/Desktop/IDS_570_final/bert_pca_clusters.png"

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} sentences.")

df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"Working with {len(df_sample)} sampled sentences.")

print("\nLoading BERT model (this may take a minute)...")
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("BERT loaded.")

def get_trade_embedding(sentence, keyword="trade"):
    """
    Returns the BERT embedding of the first trade-variant token in the sentence.
    Falls back to [CLS] embedding if no match found.
    """
    sentence = str(sentence)[:400]
    inputs = tokenizer(sentence, return_tensors="pt",
                       truncation=True, max_length=128,
                       padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.last_hidden_state[0]  # shape: (seq_len, 768)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    for i, tok in enumerate(tokens):
        if tok.startswith("trade") or tok.startswith("trad"):
            return hidden_states[i].numpy()
    
    return hidden_states[0].numpy()

print("\nGenerating BERT embeddings...")
embeddings = []
for i, row in df_sample.iterrows():
    emb = get_trade_embedding(row["sentence"])
    embeddings.append(emb)
    if i % 100 == 0:
        print(f"  {i}/1000 done...")

embeddings = np.array(embeddings)
embeddings = normalize(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

print("\nClustering into 2 groups...")
N_CLUSTERS = 2
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = km.fit_predict(embeddings)
df_sample["cluster"] = labels

print("Running PCA for visualization...")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
df_sample["pca_x"] = coords[:, 0]
df_sample["pca_y"] = coords[:, 1]

colors = ["steelblue", "darkorange", "green", "red"]
plt.figure(figsize=(10, 7))
for c in range(N_CLUSTERS):
    mask = labels == c
    plt.scatter(coords[mask, 0], coords[mask, 1],
                label=f"Cluster {c}", alpha=0.5,
                s=15, color=colors[c])
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("BERT Contextual Embeddings of 'trade'")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
plt.show()
print(f"Plot saved to: {OUTPUT_PLOT}")

print("\n=== REPRESENTATIVE EXAMPLES PER CLUSTER ===")
for c in range(N_CLUSTERS):
    cluster_df = df_sample[df_sample["cluster"] == c]
    print(f"\n--- CLUSTER {c} ({len(cluster_df)} sentences) ---")
    for _, row in cluster_df.head(4).iterrows():
        print(f"  [{row['filename']}]")
        print(f"  {str(row['sentence'])[:200]}")
        print()

df_sample.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Results saved to: {OUTPUT_CSV}")