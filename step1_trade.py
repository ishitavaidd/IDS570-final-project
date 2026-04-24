import re
import os
import csv
import matplotlib.pyplot as plt
CORPUS_FOLDER = os.path.expanduser("/Users/Ishita/Desktop/IDS_570_final/texts")
OUTPUT_CSV = "/Users/Ishita/Desktop/IDS_570_final/trade_occurrences.csv"
OUTPUT_CHART = "/Users/Ishita/Desktop/IDS_570_final/trade_distribution.png"
TRADE_PATTERN = re.compile(r'\btrad(e|es|ed|ing|er|ers)\b', re.IGNORECASE)
def get_sentence(text, match_start, match_end):
    sentence_start = text.rfind('.', 0, match_start)
    if sentence_start == -1:
        sentence_start = 0
    else:
        sentence_start += 1
    sentence_end = text.find('.', match_end)
    if sentence_end == -1:
        sentence_end = len(text)
    else:
        sentence_end += 1
    return text[sentence_start:sentence_end].strip()
 
all_results = []
file_counts = {}
 
if not os.path.exists(CORPUS_FOLDER):
    print(f"ERROR: Could not find the folder: {CORPUS_FOLDER}")
else:
    all_files = sorted([f for f in os.listdir(CORPUS_FOLDER) if f.endswith('.txt')])
    print(f"Found {len(all_files)} text files. Searching for 'trade'...\n")
 
    for filename in all_files:
        filepath = os.path.join(CORPUS_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        matches = list(TRADE_PATTERN.finditer(text))
        file_counts[filename] = len(matches)
        for match in matches:
            matched_word = match.group()
            sentence = get_sentence(text, match.start(), match.end())
            all_results.append([filename, matched_word, sentence])
 
    total_matches = len(all_results)
    files_with_matches = sum(1 for count in file_counts.values() if count > 0)
 
    print("=" * 60)
    print(f"TOTAL occurrences of 'trade' (all forms): {total_matches}")
    print(f"Files containing at least one match: {files_with_matches} / {len(all_files)}")
    print("=" * 60)
 
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 files by number of 'trade' occurrences:")
    for fname, count in sorted_files[:20]:
        print(f"  {fname}: {count}")
 
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'matched_word', 'sentence'])
        writer.writerows(all_results)
    print(f"\nResults saved to: {OUTPUT_CSV}")
 
    files_to_plot = [(f, c) for f, c in sorted_files if c > 0][:40]
    names = [f[0].replace('.txt', '') for f in files_to_plot]
    counts = [f[1] for f in files_to_plot]
 
    plt.figure(figsize=(16, 6))
    plt.bar(names, counts, color='steelblue')
    plt.xticks(rotation=90, fontsize=7)
    plt.xlabel('Document')
    plt.ylabel('Number of occurrences')
    plt.title("Distribution of 'trade' across corpus (top 40 files)")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART)
    plt.show()
    print(f"Chart saved to: {OUTPUT_CHART}")
