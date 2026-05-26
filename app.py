import streamlit as st
import pickle
import numpy as np
import os
import subprocess
import sys

st.set_page_config(
    page_title="Reddit Stress Detector",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        header {visibility: hidden !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Auto-train if model files are missing
# ------------------------------------------------------------------
MODEL_PATH      = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    with st.spinner("Model not found — training now. Please wait..."):
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True
        )
    if result.returncode != 0:
        st.error("Training failed. Please run `python main.py` manually.")
        st.code(result.stderr)
        st.stop()
    else:
        st.success("Model trained and ready!")

# ------------------------------------------------------------------
# Load model and vectorizer
# ------------------------------------------------------------------
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            m = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            v = pickle.load(f)
        return m, v
    except Exception as e:
        st.error(
            f"Failed to load model: {e}\n\n"
            "Please delete model.pkl and vectorizer.pkl, "
            "then run `python main.py` again to retrain."
        )
        st.stop()

model, vectorizer = load_model()

# ------------------------------------------------------------------
# Page Header
# ------------------------------------------------------------------
st.title("🧠 Reddit Stress Detector")
st.markdown("### Powered by TF-IDF + Multinomial Naive Bayes")
st.markdown(
    "This model was trained on **2,838 real Reddit posts** from communities like "
    "anxiety, ptsd, relationships, and more. "
    "Type or paste any text below to detect stress."
)
st.markdown("---")

# ------------------------------------------------------------------
# Text Input
# ------------------------------------------------------------------
st.subheader("Enter Text to Analyse")
user_text = st.text_area(
    label="Type or paste a message, post, or any text:",
    placeholder="Example: I haven't been sleeping well lately and everything feels overwhelming...",
    height=160,
    label_visibility="collapsed"
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    predict_clicked = st.button("🔍 Detect Stress", use_container_width=True)
with col_clear:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if predict_clicked:
    if not user_text or len(user_text.strip()) < 10:
        st.warning("Please enter at least a sentence to analyse.")
    else:
        text_vec = vectorizer.transform([user_text.strip()])
        prediction  = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        confidence  = max(probability) * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                f"⚠️ **Stress Detected**\n\n"
                f"The text shows signs of **stress or emotional distress**.\n\n"
                f"Confidence: **{confidence:.1f}%**"
            )
            st.markdown("""
**Suggestions:**
- Talk to someone you trust — a friend, family member, or counsellor
- Try deep breathing or a short walk to reset your mind
- Write down your worries to externalise them
- Consider speaking with a mental health professional
            """)
        else:
            st.success(
                f"✅ **No Stress Detected**\n\n"
                f"The text does **not** show strong signs of stress.\n\n"
                f"Confidence: **{confidence:.1f}%**"
            )
            st.markdown("""
**Keep it up:**
- Maintain your current healthy routines
- Continue reaching out and expressing yourself
- Small daily habits make a big long-term difference
            """)

        # Probability breakdown
        st.markdown("---")
        st.caption("Probability Breakdown")
        p1, p2 = st.columns(2)
        p1.metric("No Stress", f"{probability[0]*100:.1f}%")
        p2.metric("Stress",    f"{probability[1]*100:.1f}%")

        # Progress bars
        st.progress(float(probability[0]), text="No Stress probability")
        st.progress(float(probability[1]), text="Stress probability")

        # Top contributing words
        st.markdown("---")
        st.caption("Top Words Driving This Prediction")
        feature_names = vectorizer.get_feature_names_out()
        log_probs      = model.feature_log_prob_
        text_tfidf     = text_vec.toarray()[0]

        nonzero_idx = text_tfidf.nonzero()[0]
        if len(nonzero_idx) > 0:
            stress_score = (log_probs[1] - log_probs[0])[nonzero_idx]
            top_idx = nonzero_idx[np.argsort(stress_score)[::-1][:8]]
            top_words = [(feature_names[i],
                          "🔴 Stress" if log_probs[1][i] > log_probs[0][i] else "🟢 No Stress")
                         for i in top_idx]
            cols = st.columns(4)
            for j, (word, tag) in enumerate(top_words):
                cols[j % 4].markdown(f"**{word}**  \n{tag}")
        else:
            st.write("No significant keywords found.")

# ------------------------------------------------------------------
# Sidebar — About the model
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
**Dataset**  
2,838 Reddit posts from 10 communities

**Communities**
- ptsd (584)
- relationships (552)
- anxiety (503)
- domesticviolence (316)
- assistance (289)
- survivorsofabuse (245)
- homeless (168)
- almosthomeless (80)
- stress (64)
- food_pantry (37)

**Algorithm**  
Multinomial Naive Bayes  
with TF-IDF (5,000 features, 1-2 grams)

**Label**  
0 = No Stress  
1 = Stress

**Training Split**  
80% train / 20% test
    """)
    st.markdown("---")
    st.caption("For educational use only. Not a substitute for professional mental health advice.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model trained on real Reddit data. "
    "Predictions are probabilistic and for educational purposes only."
)
