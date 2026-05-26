# Stress Prediction System

A Natural Language Processing (NLP) project that detects stress in text using a Multinomial Naive Bayes classifier trained on real Reddit posts.

## Overview

This system analyses free-text input and predicts whether it contains signs of stress or not. It was trained on 2,838 Reddit posts from communities related to anxiety, PTSD, relationships, domestic violence, and more.

- **Algorithm**: Multinomial Naive Bayes
- **Text Vectorisation**: TF-IDF (5,000 features, 1–2 word grams)
- **Accuracy**: 72%
- **Dataset**: 2,838 Reddit posts, 116 columns
- **Interface**: Streamlit web app

---

## Project Structure

```
├── main.py                        # ML pipeline: train, evaluate, save model
├── app.py                         # Streamlit web app
├── stress_1776505532728.csv       # Dataset (2,838 Reddit posts)
├── model.pkl                      # Saved trained model (auto-generated)
├── vectorizer.pkl                 # Saved TF-IDF vectorizer (auto-generated)
├── .streamlit/
│   └── config.toml                # Streamlit server configuration
├── subreddit_distribution.png     # EDA chart (auto-generated)
├── text_length_distribution.png   # EDA chart (auto-generated)
├── liwc_correlation_heatmap.png   # EDA chart (auto-generated)
└── confusion_matrix.png           # Model evaluation chart (auto-generated)
```

---

## Dataset

The dataset contains Reddit posts labelled as stress (1) or no stress (0).

| Column | Description |
|---|---|
| `text` | Raw Reddit post content — main input feature |
| `label` | Target: 0 = No Stress, 1 = Stress |
| `subreddit` | Community the post came from |
| `confidence` | Human labeller confidence score (0–1) |
| `sentiment` | Pre-computed sentiment score |
| `lex_liwc_*` | 80+ LIWC linguistic and psychological features |
| `lex_dal_*` | DAL word pleasantness/activation/imagery scores |
| `syntax_*` | Readability scores (ARI, Flesch-Kincaid) |
| `social_*` | Reddit karma, upvote ratio, comment count |

**Label Distribution**
- Stress (1): 1,488 posts (52.4%)
- No Stress (0): 1,350 posts (47.6%)

**Source Communities**

| Community | Posts |
|---|---|
| ptsd | 584 |
| relationships | 552 |
| anxiety | 503 |
| domesticviolence | 316 |
| assistance | 289 |
| survivorsofabuse | 245 |
| homeless | 168 |
| almosthomeless | 80 |
| stress | 64 |
| food_pantry | 37 |

---

## How It Works

1. **Text Input** — User types or pastes any text
2. **TF-IDF Vectorisation** — Text is converted into a 5,000-number feature vector
3. **MultinomialNB Prediction** — Model outputs label 0 or 1
4. **Result Display** — Shows stress/no-stress result with confidence % and top contributing words

---

## Setup & Usage

### 1. Install dependencies (once)

```bash
pip install scikit-learn pandas numpy matplotlib seaborn streamlit
```

### 2. Train the model (once)

```bash
python main.py
```

Expected output:
```
TRAINING COMPLETE!
Final Accuracy : 72.13%
Dataset        : 2,834 Reddit posts
Algorithm      : Multinomial Naive Bayes + TF-IDF
```

This creates `model.pkl` and `vectorizer.pkl` and saves 4 EDA charts.

### 3. Run the web app

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Type any text and click **Detect Stress**.

> After the first setup, only Step 3 is needed each time.

---

## Model Pipeline (main.py)

| Step | Action |
|---|---|
| 1 | Load CSV dataset |
| 2 | Inspect shape, columns, missing values |
| 3 | Clean text — drop nulls, strip whitespace |
| 4 | Generate 3 EDA charts (saved as PNG) |
| 5 | Split: 80% train / 20% test |
| 6 | Fit TF-IDF vectoriser on training text |
| 7 | Train MultinomialNB model |
| 8 | Evaluate: accuracy, confusion matrix, classification report |
| 9 | Save `model.pkl` and `vectorizer.pkl` |

---

## Web App (app.py)

| Step | Action |
|---|---|
| 1 | Check for model files — auto-trains if missing |
| 2 | Load `model.pkl` and `vectorizer.pkl` |
| 3 | Display text input area |
| 4 | On predict: vectorise → classify → show result |
| 5 | Show confidence %, probability bars, and top keywords |

---

## Requirements

| Package | Purpose |
|---|---|
| `pandas` | Load and clean the CSV dataset |
| `numpy` | Array operations |
| `scikit-learn` | TfidfVectorizer, MultinomialNB, metrics |
| `matplotlib` | Save EDA chart images |
| `seaborn` | Styled charts (heatmap, histogram, bar) |
| `streamlit` | Web UI with text input and result display |
| `pickle` | Save and load model and vectoriser |

---

## Notes

- The model uses only the `text` column for prediction via TF-IDF
- Adding LIWC/DAL features (already in the CSV) could improve accuracy further
- 72% accuracy reflects the natural complexity of stress language (sarcasm, context, tone)

---

> For educational purposes only. Not a substitute for professional mental health advice.
