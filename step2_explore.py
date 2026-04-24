import pandas as pd
import re

df = pd.read_csv('/Users/Ishita/Desktop/IDS_570_final/trade_ner_results.csv')

all_entities = []
for _, row in df.iterrows():
    if pd.isna(row['entities']):
        continue
    parts = str(row['entities']).split(' | ')
    for part in parts:
        match = re.match(r'^(.+)\s+\[([A-Z_]+)\]$', part.strip())
        if match:
            text = match.group(1).strip().lower()
            label = match.group(2)
            all_entities.append({
                'text': text,
                'label': label,
                'filename': row['filename'],
                'sentence': row['sentence']
            })

ent_df = pd.DataFrame(all_entities)

def show_random_examples(entity_type, n=5):
    subset = ent_df[ent_df['label'] == entity_type]
    sample = subset.drop_duplicates(subset='sentence').sample(n, random_state=42)
    print(f"\n{'='*60}")
    print(f"ENTITY TYPE: {entity_type}")
    print(f"{'='*60}\n")
    for _, row in sample.iterrows():
        print(f"Entity: {row['text']} | File: {row['filename']}")
        print(f"Sentence: {row['sentence'][:400]}")
        print()

# Run for three different entity types to compare
show_random_examples('GPE')
show_random_examples('ORG')
show_random_examples('PERSON')