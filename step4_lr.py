import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

INPUT_CSV   = "/Users/Ishita/Desktop/IDS_570_final/trade_occurrences.csv"
OUTPUT_CSV  = "/Users/Ishita/Desktop/IDS_570_final/trade_classified.csv"
OUTPUT_CM   = "/Users/Ishita/Desktop/IDS_570_final/confusion_matrix.png"
OUTPUT_COEF = "/Users/Ishita/Desktop/IDS_570_final/top_features.png"

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} sentences.")

POLICY_KEYWORDS = [
    r'\bmerchant\b', r'\bcommerce\b', r'\bbalance\b', r'\bnation\b',
    r'\bkingdom\b', r'\bengland\b', r'\bholland\b', r'\bfrance\b',
    r'\bportugal\b', r'\bindia\b', r'\bexport\b', r'\bimport\b',
    r'\bcustom\b', r'\bduty\b', r'\bduties\b', r'\bmonopol\b',
    r'\bstaple\b', r'\bcommodit\b', r'\bwealth\b', r'\bcompan\b',
    r'\bforeign\b', r'\bnational\b', r'\bship\b', r'\bport\b',
    r'\bmarket\b', r'\bprofite\b', r'\bprofit\b', r'\bexchang\b',
    r'\btraffi\b', r'\bnavig\b', r'\beast ind\b', r'\bwest ind\b',
]

OCCUPATION_KEYWORDS = [
    r'\btradesman\b', r'\btradesmen\b', r'\bcraftsman\b', r'\bartisan\b',
    r'\bguild\b', r'\bhandicraft\b', r'\bmechanic\b', r'\blabou\b',
    r'\blabor\b', r'\bcalling\b', r'\bvocation\b', r'\boccupation\b',
    r'\bprofession\b', r'\bprinter\b', r'\btailor\b', r'\bbutcher\b',
    r'\bshoemaker\b', r'\bcarpenter\b', r'\bsmith\b', r'\bweaver\b',
    r'\bapprentice\b', r'\bcorpora\b', r'\bincorporat\b',
    r'\blive by\b', r'\blivelihood\b', r'\bundone\b', r'\bsinne\b',
    r'\bvictualer\b', r'\blabouring man\b', r'\bhandy\b',
]

def weak_label(sentence):
    s = str(sentence).lower()
    policy_score    = sum(1 for p in POLICY_KEYWORDS     if re.search(p, s))
    occupation_score = sum(1 for p in OCCUPATION_KEYWORDS if re.search(p, s))
    if policy_score > occupation_score:
        return 0   # policy/commerce
    elif occupation_score > policy_score:
        return 1   # occupation/craft
    else:
        return -1  # ambiguous — exclude

df["label"] = df["sentence"].apply(weak_label)

print("\n=== WEAK LABEL DISTRIBUTION ===")
print(df["label"].value_counts())

labeled = df[df["label"] != -1].copy()
print(f"\nUsable labeled sentences: {len(labeled)}")
print(f"  Policy/Commerce (0): {(labeled['label']==0).sum()}")
print(f"  Occupation/Craft (1): {(labeled['label']==1).sum()}")

print("\n=== SAMPLE LABELED SENTENCES ===")
print("\n-- POLICY/COMMERCE (label=0) --")
for _, row in labeled[labeled["label"]==0].head(4).iterrows():
    print(f"  {str(row['sentence'])[:200]}\n")

print("\n-- OCCUPATION/CRAFT (label=1) --")
for _, row in labeled[labeled["label"]==1].head(4).iterrows():
    print(f"  {str(row['sentence'])[:200]}\n")

print("\n=== TRAINING LOGISTIC REGRESSION ===")

X = labeled["sentence"].astype(str)
y = labeled["label"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)

X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {X_train.shape[0]} sentences, testing on {X_test.shape[0]}")

clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred,
      target_names=["Policy/Commerce", "Occupation/Craft"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Policy/Commerce", "Occupation"],
            yticklabels=["Policy/Commerce", "Occupation"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_CM)
plt.show()
print(f"Confusion matrix saved to: {OUTPUT_CM}")

feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]

top_n = 20
top_policy_idx    = np.argsort(coefs)[::-1][:top_n]
top_occupation_idx = np.argsort(coefs)[:top_n]

top_policy_words    = [feature_names[i] for i in top_policy_idx]
top_policy_vals     = [coefs[i] for i in top_policy_idx]
top_occupation_words = [feature_names[i] for i in top_occupation_idx]
top_occupation_vals  = [coefs[i] for i in top_occupation_idx]

print("\n=== TOP FEATURES: POLICY/COMMERCE ===")
for w, v in zip(top_policy_words, top_policy_vals):
    print(f"  {w:30s}: {v:.3f}")

print("\n=== TOP FEATURES: OCCUPATION ===")
for w, v in zip(top_occupation_words, top_occupation_vals):
    print(f"  {w:30s}: {v:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.barh(top_policy_words[::-1], top_policy_vals[::-1], color='steelblue')
ax1.set_title("Top Features → Occupation")
ax1.set_xlabel("Coefficient")

ax2.barh(top_occupation_words, [abs(v) for v in top_occupation_vals], color='darkorange')
ax2.set_title("Top Features → Policy/Commerce")
ax2.set_xlabel("Coefficient")

plt.suptitle("Most Distinguishable Features")
plt.tight_layout()
plt.savefig(OUTPUT_COEF)
plt.show()
print(f"Feature chart saved to: {OUTPUT_COEF}")

print("\n=== APPLYING MODEL TO FULL CORPUS ===")
unlabeled = df[df["label"] == -1].copy()
full_tfidf = vectorizer.transform(df["sentence"].astype(str))
df["predicted_label"] = clf.predict(full_tfidf)
df["predicted_label_name"] = df["predicted_label"].map(
    {0: "Policy/Commerce", 1: "Occupation/Craft"}
)

print("\nFull corpus predicted distribution:")
print(df["predicted_label_name"].value_counts())

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nFull results saved to: {OUTPUT_CSV}")