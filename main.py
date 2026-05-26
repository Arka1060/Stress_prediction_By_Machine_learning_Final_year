import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

print("=" * 65)
print("   REDDIT STRESS PREDICTION SYSTEM - NLP + NAIVE BAYES")
print("=" * 65)

# ------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------
print("\n[1] Loading dataset...")
import glob, os
# Prefer attached_assets/ first; pick the CSV with the most columns (Reddit = 116 cols)
csv_files = glob.glob("attached_assets/stress_*.csv") + glob.glob("stress_*.csv")
if not csv_files:
    raise FileNotFoundError(
        "CSV file not found. Place your stress_*.csv file in the same folder as main.py and try again."
    )
import pandas as _pd_tmp
csv_path = max(csv_files, key=lambda f: len(_pd_tmp.read_csv(f, nrows=1).columns))
df = pd.read_csv(csv_path)
print(f"    Loaded : {os.path.basename(csv_path)}")
print(f"    Shape  : {df.shape[0]} rows x {df.shape[1]} columns")

# ------------------------------------------------------------------
# 2. Inspect Data
# ------------------------------------------------------------------
print("\n[2] Inspecting data...")
print(f"    Columns (first 10): {list(df.columns[:10])}")
print(f"    Missing values in 'text'  : {df['text'].isnull().sum()}")
print(f"    Missing values in 'label' : {df['label'].isnull().sum()}")
print(f"    Label distribution:")
print(df['label'].value_counts().to_string(header=False))

# ------------------------------------------------------------------
# 3. Clean & Select Relevant Columns
# ------------------------------------------------------------------
print("\n[3] Cleaning data...")
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 10]
df['label'] = df['label'].astype(int)
print(f"    Clean dataset size: {len(df)} rows")
print(f"    Label 0 (No Stress): {(df['label']==0).sum()}")
print(f"    Label 1 (Stress)   : {(df['label']==1).sum()}")

# ------------------------------------------------------------------
# 4. EDA — Charts
# ------------------------------------------------------------------
print("\n[4] Generating EDA charts...")

# Chart 1: Posts per Subreddit
plt.figure(figsize=(10, 5))
counts = df['subreddit'].value_counts()
colors = ['#e74c3c' if sr in ['ptsd','anxiety','domesticviolence','survivorsofabuse','stress']
          else '#3498db' for sr in counts.index]
bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8)
plt.title('Reddit Posts per Subreddit', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Subreddit', fontsize=11)
plt.ylabel('Number of Posts', fontsize=11)
plt.xticks(rotation=30, ha='right')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
from matplotlib.patches import Patch
legend = [Patch(color='#e74c3c', label='Stress-related'),
          Patch(color='#3498db', label='Non-stress')]
plt.legend(handles=legend)
plt.tight_layout()
plt.savefig('subreddit_distribution.png', dpi=100)
plt.close()
print("    Saved: subreddit_distribution.png")

# Chart 2: Text Length Distribution by Label
df['word_count'] = df['text'].str.split().str.len()
plt.figure(figsize=(10, 5))
for label, color, name in [(0, '#2ecc71', 'No Stress'), (1, '#e74c3c', 'Stress')]:
    subset = df[df['label'] == label]['word_count']
    plt.hist(subset, bins=40, alpha=0.6, color=color, label=f'{name} (n={len(subset)})', edgecolor='white')
plt.title('Post Word Count Distribution by Stress Label', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Word Count', fontsize=11)
plt.ylabel('Number of Posts', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('text_length_distribution.png', dpi=100)
plt.close()
print("    Saved: text_length_distribution.png")

# Chart 3: LIWC features correlation with label
liwc_cols = [c for c in df.columns if c.startswith('lex_liwc_')][:20]
if liwc_cols:
    corr_vals = df[liwc_cols + ['label']].corr()['label'].drop('label').sort_values()
    plt.figure(figsize=(10, 7))
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in corr_vals.values]
    plt.barh(corr_vals.index, corr_vals.values, color=colors, edgecolor='white')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('LIWC Feature Correlation with Stress Label', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Pearson Correlation', fontsize=11)
    plt.tight_layout()
    plt.savefig('liwc_correlation_heatmap.png', dpi=100)
    plt.close()
    print("    Saved: liwc_correlation_heatmap.png")

# ------------------------------------------------------------------
# 5. Prepare X (text) and y (label)
# ------------------------------------------------------------------
print("\n[5] Preparing features and labels...")
X = df['text'].tolist()
y = df['label'].to_numpy(dtype=int)
print(f"    Total samples: {len(X)}")

# ------------------------------------------------------------------
# 6. Train/Test Split
# ------------------------------------------------------------------
print("\n[6] Splitting dataset (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# ------------------------------------------------------------------
# 7. TF-IDF Vectorization
# ------------------------------------------------------------------
print("\n[7] Applying TF-IDF vectorization (max_features=5000)...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"    Vocabulary size   : {len(vectorizer.vocabulary_)}")
print(f"    Training matrix   : {X_train_vec.shape}")
print(f"    Test matrix       : {X_test_vec.shape}")

# ------------------------------------------------------------------
# 8. Train MultinomialNB Model
# ------------------------------------------------------------------
print("\n[8] Training Multinomial Naive Bayes model...")
model = MultinomialNB(alpha=0.5)
model.fit(X_train_vec, y_train)
print("    Model trained successfully.")

# ------------------------------------------------------------------
# 9. Evaluate Model
# ------------------------------------------------------------------
print("\n[9] Evaluating model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n    Accuracy : {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\n    Confusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Stress', 'Stress'],
            yticklabels=['No Stress', 'Stress'])
plt.title('Confusion Matrix', fontsize=13, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.close()
print("    Saved: confusion_matrix.png")

print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))

# Top stress / non-stress words
feature_names = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_
stress_scores = log_probs[1] - log_probs[0]
top_stress   = [feature_names[i] for i in stress_scores.argsort()[-10:][::-1]]
top_nostress = [feature_names[i] for i in stress_scores.argsort()[:10]]
print(f"\n    Top 10 stress words     : {top_stress}")
print(f"    Top 10 non-stress words : {top_nostress}")

# ------------------------------------------------------------------
# 10. Save Model and Vectorizer
# ------------------------------------------------------------------
print("\n[10] Saving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("    Saved: model.pkl")
print("    Saved: vectorizer.pkl")

print("\n" + "=" * 65)
print("   TRAINING COMPLETE!")
print(f"   Final Accuracy : {accuracy * 100:.2f}%")
print(f"   Dataset        : {len(df)} Reddit posts")
print(f"   Algorithm      : Multinomial Naive Bayes + TF-IDF")
print("=" * 65)