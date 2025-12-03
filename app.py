# app.py
# FIN545 – SERI Scam Exposure Risk Index (Streamlit app)
# Coded by: Shravani Sawant 
# simple, clean UI using the trained LR (as just a prototype) + TF-IDF model
# and the behavioural SERI logic from the notebook.

import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. Page config MUST be first Streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title="SERI – Scam Exposure Risk Index",
    layout="wide",
)

# ---------------------------------------------------------
# 1a. Light styling (white background, subtle tweaks)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    h1, h2, h3, h4 {
        color: #111827;
    }

    /* Primary button */
    .stButton>button {
        background: linear-gradient(90deg, #f97316, #ec4899);
        border-radius: 999px;
        color: white;
        border: none;
        padding: 0.45rem 1.6rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        filter: brightness(1.05);
    }

    /* Text area */
    textarea {
        border-radius: 0.7rem !important;
        border: 1px solid #d1d5db !important;
        background-color: #f9fafb !important;
        color: #111827 !important;
    }

    /* Tables / dataframes */
    .stTable, .stDataFrame {
        border-radius: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. Load model, vectorizer and SERI metadata (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load TF-IDF vectorizer, classifier, and SERI lexicons/weights."""
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("phishing_classifier.joblib")

    with open("seri_lexicons.json", "r") as f:
        lexicons = json.load(f)

    with open("seri_weights.json", "r") as f:
        weights = json.load(f)

    return vectorizer, model, lexicons, weights


vectorizer, model, LEXICONS, WEIGHTS = load_artifacts()

URGENCY_WORDS = LEXICONS.get("URGENCY_WORDS", [])
AUTHORITY_WORDS = LEXICONS.get("AUTHORITY_WORDS", [])
REWARD_WORDS = LEXICONS.get("REWARD_WORDS", [])
FEAR_WORDS = LEXICONS.get("FEAR_WORDS", [])

ALPHA = WEIGHTS.get("ALPHA", 0.35)   # behavioural
BETA = WEIGHTS.get("BETA", 0.00)     # technical (0 for now)
GAMMA = WEIGHTS.get("GAMMA", 0.45)   # classifier risk
DELTA = WEIGHTS.get("DELTA", 0.20)   # uncertainty


# ---------------------------------------------------------
# 3. Helper functions – behaviour features + SERI logic
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Basic normalisation: strip spaces, collapse whitespace."""
    t = str(text).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def score_keyword_group(text: str, keywords: list[str]) -> float:
    """
    Return a normalised score in [0, 1] based on how many
    of the group keywords appear in the message.
    """
    t = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in t)
    hits_capped = min(hits, 3)  # cap so long messages don't explode
    return hits_capped / 3.0


def extract_behaviour_features(text: str) -> dict:
    """Compute urgency, authority, reward and fear scores."""
    return {
        "urgency_score": score_keyword_group(text, URGENCY_WORDS),
        "authority_score": score_keyword_group(text, AUTHORITY_WORDS),
        "reward_score": score_keyword_group(text, REWARD_WORDS),
        "fear_score": score_keyword_group(text, FEAR_WORDS),
    }


def compute_seri_row(beh_feats: dict, p_scam: float, max_class_prob: float) -> float:
    """
    Compute SERI for a single message, matching the notebook logic.

    R_b = behavioural risk (average of the four scores)
    R_t = technical risk (0 for now)
    R_c = classifier risk (p_scam)
    U   = uncertainty (1 - max_class_prob)
    """
    R_b = (
        beh_feats["urgency_score"]
        + beh_feats["authority_score"]
        + beh_feats["reward_score"]
        + beh_feats["fear_score"]
    ) / 4.0

    R_t = 0.0
    R_c = p_scam
    U = 1.0 - max_class_prob

    seri_raw = ALPHA * R_b + BETA * R_t + GAMMA * R_c + DELTA * U
    seri = 100.0 * float(np.clip(seri_raw, 0.0, 1.0))
    return seri


def risk_band(seri: float) -> str:
    """Convert SERI score into a qualitative band."""
    if seri < 25:
        return "Low"
    elif seri < 50:
        return "Medium"
    elif seri < 75:
        return "High"
    else:
        return "Critical"


def analyse_message(text: str) -> dict:
    """
    Full pipeline for a single message:
      - clean text
      - TF-IDF transform
      - LR predict + probabilities
      - behavioural features
      - SERI score + band
    """
    msg = clean_text(text)

    # ---- 1) Vectorise with the *fitted* TF-IDF vectorizer ----
    try:
        X_vec = vectorizer.transform([msg])
    except ValueError as e:
        # This is where we used to see "idf vector is not fitted"
        raise RuntimeError(
            f"TF-IDF vectoriser problem: {e}. "
            "Make sure tfidf_vectorizer.joblib comes from FIN545_Model.ipynb "
            "after calling fit_transform on the training data."
        )

    # ---- 2) Classifier prediction ----
    proba = model.predict_proba(X_vec)[0]
    p_safe = float(proba[0])
    p_scam = float(proba[1])
    max_prob = float(proba.max())
    pred_label = "Phishing" if p_scam >= 0.5 else "Safe"

    # ---- 3) Behavioural features ----
    beh = extract_behaviour_features(msg)

    # ---- 4) SERI ----
    seri_score = compute_seri_row(beh, p_scam=p_scam, max_class_prob=max_prob)
    band = risk_band(seri_score)

    return {
        "clean_text": msg,
        "prediction": pred_label,
        "p_scam": p_scam,
        "p_safe": p_safe,
        "max_prob": max_prob,
        "seri": seri_score,
        "risk_band": band,
        "behaviour": beh,
    }


# ---------------------------------------------------------
# 4. Session history
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts


# ---------------------------------------------------------
# 5. UI layout
# ---------------------------------------------------------
st.title("SERI – Scam Exposure Risk Index")

st.markdown(
    """
    FIN-545 Research Prototype – **Behavioural + ML-based Scam Risk Scoring**

    Paste a suspicious **email, SMS, or chat message** below.  
    The system will:

    - Predict whether it looks like **phishing** or **safe**
    - Estimate the **Scam Exposure Risk Index (SERI)** from 0–100
    - Break down **behavioural cues**: urgency, authority, reward, fear  

    _Results are generated by a logistic regression model trained on our curated FIN-545 FinTech scam dataset._
    """
)

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Message text")

    example_placeholder = (
        "e.g., Your brokerage account requires confirmation of tax documents. "
        "Review the form to prevent delays during processing."
    )

    message_text = st.text_area(
        "Paste one message per analysis:",
        value="",
        height=180,
        placeholder=example_placeholder,
    )

    analyse_button = st.button("Analyse Message", type="primary")

with col_right:
    st.subheader("How to read the output")
    st.markdown(
        """
        - **Prediction**: final phishing vs safe decision.
        - **Scam probability**: model's estimated chance that it is phishing.
        - **SERI (0–100)**:
            - 0–25 → Low  
            - 25–50 → Medium  
            - 50–75 → High  
            - 75–100 → Critical  
        - Behavioural scores are in **[0, 1]** – higher means stronger cue.

        _This is a research prototype and not financial advice._
        """
    )

st.markdown("---")

# ---------------------------------------------------------
# 6. Run analysis when the user clicks
# ---------------------------------------------------------
if analyse_button:
    if not message_text.strip():
        st.warning("Please paste a message before clicking **Analyse Message**.")
    else:
        try:
            result = analyse_message(message_text)
        except Exception as e:
            st.error(f"Something went wrong while analysing the message: {e}")
        else:
            # --- Top summary cards ---
            col1, col2, col3 = st.columns(3)

            with col1:
                if result["prediction"] == "Phishing":
                    st.error(f"Prediction: **{result['prediction']}**")
                else:
                    st.success(f"Prediction: **{result['prediction']}**")

                st.markdown(
                    f"*Model scam probability:* **{result['p_scam'] * 100:.1f}%**"
                )

            with col2:
                st.metric(
                    label="SERI score (0–100)",
                    value=f"{result['seri']:.1f}",
                )

            with col3:
                band = result["risk_band"]
                emoji = {
                    "Low": "🟢",
                    "Medium": "🟡",
                    "High": "🟠",
                    "Critical": "🔴",
                }.get(band, "⚪️")
                st.markdown(f"**Risk band:** {emoji} **{band}**")

            st.markdown("### Behavioural cues")
            beh = result["behaviour"]
            beh_df = pd.DataFrame(
                {
                    "Cue": ["Urgency", "Authority", "Reward", "Fear"],
                    "Score (0–1)": [
                        beh["urgency_score"],
                        beh["authority_score"],
                        beh["reward_score"],
                        beh["fear_score"],
                    ],
                }
            )
            st.table(beh_df.style.format({"Score (0–1)": "{:.2f}"}))

            # Save to session history
            st.session_state.history.append(
                {
                    "Message snippet": result["clean_text"][:100] + (
                        "..." if len(result["clean_text"]) > 100 else ""
                    ),
                    "Prediction": result["prediction"],
                    "p_scam (%)": round(result["p_scam"] * 100, 1),
                    "SERI": round(result["seri"], 1),
                    "Risk band": result["risk_band"],
                }
            )

# ---------------------------------------------------------
# 7. History table
# ---------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("Analysis history (this session)")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

# ---------------------------------------------------------
# 8. Footer – copyright + credits
# ---------------------------------------------------------
st.markdown(
    """
    <hr style="margin-top: 40px;">
    <div style='text-align: center; color: #6b7280; font-size: 13px;'>
        © 2025 <b>Shravani Sawant</b>. <br> All rights reserved.<br>
        These results are generated by a logistic regression model trained on
        a custom FIN-545 FinTech scam dataset created for research purposes.<br>
        Built & coded by <b>@shravanips</b> – behavioural & ML-based scam risk analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

# Running:
# In terminal:
# conda activate tf311
# streamlit run app.py
