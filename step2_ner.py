import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import csv
import os

INPUT_CSV  = "/Users/Ishita/Desktop/IDS_570_final/trade_occurrences.csv"
OUTPUT_NER = "/Users/Ishita/Desktop/IDS_570_final/trade_ner_results.csv"
OUTPUT_FREQ_CHART = "/Users/Ishita/Desktop/IDS_570_final/ner_entity_freq.png"
OUTPUT_GPE_CHART  = "/Users/Ishita/Desktop/IDS_570_final/ner_top_gpe.png"

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} sentences.")

rows = []          
entity_counter = Counter()
gpe_counter    = Counter()
person_counter = Counter()
org_counter    = Counter()

for i, row in df.iterrows():
    sentence = str(row['sentence'])
    filename = row['filename']
    doc = nlp(sentence[:512])          
    
    ents_in_sentence = []
    for ent in doc.ents:
        entity_counter[ent.label_] += 1
        ents_in_sentence.append(f"{ent.text} [{ent.label_}]")
        
        if ent.label_ in ("GPE", "LOC"):
            gpe_counter[ent.text.lower()] += 1
        elif ent.label_ == "PERSON":
            person_counter[ent.text.lower()] += 1
        elif ent.label_ == "ORG":
            org_counter[ent.text.lower()] += 1
    
    rows.append({
        "filename":  filename,
        "matched_word": row['matched_word'],
        "sentence":  sentence,
        "entities":  " | ".join(ents_in_sentence)
    })

    if i % 500 == 0:
        print(f"  processed {i}/{len(df)} sentences...")

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_NER, index=False, encoding="utf-8")
print(f"\nNER results saved to: {OUTPUT_NER}")

print("\n=== ENTITY TYPE FREQUENCY ===")
for label, count in entity_counter.most_common():
    print(f"  {label:10s}: {count}")

print("\n=== TOP 20 GPE/LOC ===")
for name, count in gpe_counter.most_common(20):
    print(f"  {name:30s}: {count}")

print("\n=== TOP 20 PERSONS ===")
for name, count in person_counter.most_common(20):
    print(f"  {name:30s}: {count}")

print("\n=== TOP 20 ORGS ===")
for name, count in org_counter.most_common(20):
    print(f"  {name:30s}: {count}")
print("\n=== EXAMPLE SENTENCES: trade + GPE/LOC ===")
gpe_examples = out_df[out_df['entities'].str.contains(r'\[GPE\]|\[LOC\]', na=False)]
for _, ex in gpe_examples.head(5).iterrows():
    print(f"\n  File: {ex['filename']}")
    print(f"  Sentence: {ex['sentence'][:200]}")
    print(f"  Entities: {ex['entities']}")
labels  = [x[0] for x in entity_counter.most_common(10)]
counts  = [x[1] for x in entity_counter.most_common(10)]
plt.figure(figsize=(10, 5))
plt.bar(labels, counts, color='steelblue')
plt.xlabel("Entity Type")
plt.ylabel("Count")
plt.title("Named Entity Types Co-occurring with 'trade'")
plt.tight_layout()
plt.savefig(OUTPUT_FREQ_CHART)
plt.show()
print(f"Chart saved to: {OUTPUT_FREQ_CHART}")

top_gpe   = gpe_counter.most_common(20)
gpe_names = [x[0] for x in top_gpe]
gpe_vals  = [x[1] for x in top_gpe]
plt.figure(figsize=(12, 5))
plt.bar(gpe_names, gpe_vals, color='darkorange')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.xlabel("Place Name (GPE/LOC)")
plt.ylabel("Count")
plt.title("Top 20 Places Co-occurring with 'trade'")
plt.tight_layout()
plt.savefig(OUTPUT_GPE_CHART)
plt.show()
print(f"Chart saved to: {OUTPUT_GPE_CHART}")


