import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Match Outcome Predictor",
    page_icon="💘",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #f0eaff; }
    h1, h2, h3 { color: #e8b4ff; }
    .stSlider > div > div > div > div { background: #a855f7; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #c4b5fd; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #e8b4ff; font-size: 2rem; }
    div[data-testid="stMetricLabel"] { color: #a78bfa; }
    .prediction-box {
        background: linear-gradient(135deg, #1e1033 0%, #2d1b4e 100%);
        border: 1px solid #7c3aed;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .outcome-label {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f0abfc;
        letter-spacing: 0.05em;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #a78bfa;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .section-divider {
        border: none;
        border-top: 1px solid #3b1f6e;
        margin: 2rem 0;
    }
    .tip-box {
        background: #1a1030;
        border-left: 3px solid #7c3aed;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #c4b5fd;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper required by the pickled pipeline ────────────────────────────────
def split_interest_tags(text):
    """Split the comma-separated interest_tags column into clean tag tokens."""
    return [tag.strip() for tag in str(text).split(",") if tag.strip()]

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("best_model.pkl")
    le       = joblib.load("label_encoder.pkl")
    return model, le

try:
    model, le = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️  Model files not found. Make sure `best_model.pkl` and `label_encoder.pkl` are in the same directory.")
    model_loaded = False
    st.stop()


# ── Constants matching the training dataset ────────────────────────────────
GENDER_OPTIONS           = ["Male", "Female", "Non-binary", "Genderfluid", "Prefer Not to Say"]
ORIENTATION_OPTIONS      = ["Straight", "Gay", "Bisexual", "Pansexual", "Asexual"]
LOCATION_OPTIONS         = ["Urban", "Suburban", "Metro", "Rural", "Remote Area"]
INCOME_OPTIONS           = ["Very Low", "Low", "Middle", "Upper-Middle", "High"]
EDUCATION_OPTIONS        = ["No Formal Education", "High School", "Bachelor's", "Master's", "PhD", "Postdoc"]
APP_USAGE_LABEL_OPTIONS  = ["Light User", "Moderate User", "Heavy User", "Extreme User"]
SWIPE_LABEL_OPTIONS      = ["Very Selective", "Selective", "Moderate", "Optimistic", "Very Optimistic"]
SWIPE_TOD_OPTIONS        = ["Morning", "Afternoon", "Evening", "Late Night"]
BODY_TYPE_OPTIONS        = ["Slim", "Athletic", "Average", "Curvy", "Plus Size", "Muscular"]
ZODIAC_OPTIONS           = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo",
                             "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
RELATIONSHIP_OPTIONS     = ["Serious Relationship", "Casual Dating", "Hookups",
                             "Friends Only", "Networking", "Exploring"]
INTEREST_TAG_OPTIONS     = ["Fitness", "Politics", "Traveling", "Languages", "Fashion",
                             "Parenting", "Movies", "Reading", "DIY", "Coding", "Podcasts",
                             "History", "Clubbing", "Cars", "Gaming", "Music", "Art",
                             "Cooking", "Hiking", "Photography"]

OUTCOME_EMOJI = {
    "Blocked":            "🚫",
    "Catfished":          "🎭",
    "Chat Ignored":       "🔇",
    "Date Happened":      "📅",
    "Ghosted":            "👻",
    "Instant Match":      "⚡",
    "Mutual Match":       "💞",
    "No Action":          "😶",
    "One-sided Like":     "💙",
    "Relationship Formed":"💍",
}

OUTCOME_COLOR = {
    "Blocked":            "#ef4444",
    "Catfished":          "#f97316",
    "Chat Ignored":       "#64748b",
    "Date Happened":      "#22c55e",
    "Ghosted":            "#94a3b8",
    "Instant Match":      "#facc15",
    "Mutual Match":       "#a855f7",
    "No Action":          "#475569",
    "One-sided Like":     "#38bdf8",
    "Relationship Formed":"#ec4899",
}


# ── Header ─────────────────────────────────────────────────────────────────
st.title("💘 Match Outcome Predictor")
st.markdown("Fill in a dating profile below and see what the model predicts will happen.")
st.markdown('<div class="tip-box">💡 All fields mirror the training data — the more accurate your inputs, the more meaningful the prediction.</div>', unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── Input form ─────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    st.subheader("👤  Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.slider("Age", 18, 60, 28)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
    with c2:
        height_cm  = st.slider("Height (cm)", 145, 210, 170)
        weight_kg  = st.slider("Weight (kg)", 40.0, 130.0, 70.0, step=0.5)
    with c3:
        body_type   = st.selectbox("Body type", BODY_TYPE_OPTIONS)
        zodiac_sign = st.selectbox("Zodiac sign", ZODIAC_OPTIONS)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🌍  Background")
    c4, c5, c6 = st.columns(3)
    with c4:
        sexual_orientation = st.selectbox("Sexual orientation", ORIENTATION_OPTIONS)
        location_type      = st.selectbox("Location type", LOCATION_OPTIONS)
    with c5:
        income_bracket  = st.selectbox("Income bracket", INCOME_OPTIONS)
        education_level = st.selectbox("Education level", EDUCATION_OPTIONS)
    with c6:
        relationship_intent = st.selectbox("Relationship intent", RELATIONSHIP_OPTIONS)
        interest_tags       = st.multiselect("Interest tags (pick 1–5)", INTEREST_TAG_OPTIONS,
                                             default=["Fitness", "Traveling"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("📱  App Behaviour")
    c7, c8, c9 = st.columns(3)
    with c7:
        app_usage_time_min   = st.slider("Daily app usage (min)", 0, 300, 90)
        app_usage_time_label = st.selectbox("Usage category", APP_USAGE_LABEL_OPTIONS, index=1)
    with c8:
        swipe_right_ratio  = st.slider("Swipe-right ratio", 0.0, 1.0, 0.45, step=0.01)
        swipe_right_label  = st.selectbox("Swipe style", SWIPE_LABEL_OPTIONS, index=2)
        swipe_time_of_day  = st.selectbox("Peak swipe time", SWIPE_TOD_OPTIONS, index=2)
    with c9:
        last_active_hour = st.slider("Last active hour (0–23)", 0, 23, 21)
        profile_pics_count = st.slider("Profile photos", 1, 6, 3)
        bio_length         = st.slider("Bio length (chars)", 0, 500, 150)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("💬  Engagement")
    c10, c11, c12 = st.columns(3)
    with c10:
        likes_received    = st.slider("Likes received", 0, 200, 80)
        mutual_matches    = st.slider("Mutual matches", 0, 30, 10)
    with c11:
        message_sent_count = st.slider("Messages sent", 0, 100, 25)
        emoji_usage_rate   = st.slider("Emoji usage rate", 0.0, 1.0, 0.30, step=0.01)
    with c12:
        st.markdown("&nbsp;")  # spacer

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  Predict Match Outcome")


# ── Prediction ─────────────────────────────────────────────────────────────
if submitted:
    interest_str = ", ".join(interest_tags) if interest_tags else "Fitness"

    input_data = pd.DataFrame([{
        "gender":               gender,
        "sexual_orientation":   sexual_orientation,
        "location_type":        location_type,
        "income_bracket":       income_bracket,
        "education_level":      education_level,
        "interest_tags":        interest_str,
        "app_usage_time_min":   app_usage_time_min,
        "app_usage_time_label": app_usage_time_label,
        "swipe_right_ratio":    swipe_right_ratio,
        "swipe_right_label":    swipe_right_label,
        "likes_received":       likes_received,
        "mutual_matches":       mutual_matches,
        "profile_pics_count":   profile_pics_count,
        "bio_length":           bio_length,
        "message_sent_count":   message_sent_count,
        "emoji_usage_rate":     emoji_usage_rate,
        "last_active_hour":     last_active_hour,
        "swipe_time_of_day":    swipe_time_of_day,
        "age":                  age,
        "height_cm":            height_cm,
        "weight_kg":            weight_kg,
        "zodiac_sign":          zodiac_sign,
        "body_type":            body_type,
        "relationship_intent":  relationship_intent,
    }])

    # ── Get prediction + probabilities ──────────────────────────────────
    pred_encoded = model.predict(input_data)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]
    proba        = model.predict_proba(input_data)[0]
    class_labels = le.inverse_transform(range(len(proba)))

    confidence   = proba[pred_encoded] * 100
    emoji        = OUTCOME_EMOJI.get(pred_label, "🎯")
    color        = OUTCOME_COLOR.get(pred_label, "#a855f7")

    # ── Top result card ─────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🎯  Prediction Result")

    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size:3.5rem; margin-bottom:0.5rem">{emoji}</div>
            <div class="outcome-label" style="color:{color}">{pred_label}</div>
            <div class="confidence-text">Model confidence: <strong>{confidence:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Top-3 alternatives
        top3_idx    = np.argsort(proba)[::-1][:3]
        top3_labels = [class_labels[i] for i in top3_idx]
        top3_probs  = [proba[i] * 100 for i in top3_idx]

        st.markdown("**Top 3 likely outcomes**")
        for rank, (lbl, pct) in enumerate(zip(top3_labels, top3_probs), 1):
            bar_color = OUTCOME_COLOR.get(lbl, "#a855f7")
            st.markdown(f"""
            <div style="margin:6px 0">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                    <span style="color:#e8b4ff;font-size:0.9rem">{rank}. {OUTCOME_EMOJI.get(lbl,'')} {lbl}</span>
                    <span style="color:#a78bfa;font-size:0.9rem">{pct:.1f}%</span>
                </div>
                <div style="background:#2d1b4e;border-radius:6px;height:8px;width:100%">
                    <div style="background:{bar_color};width:{min(pct,100):.1f}%;height:8px;border-radius:6px;transition:width 0.5s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        # Full probability chart
        sorted_idx    = np.argsort(proba)
        sorted_labels = [class_labels[i] for i in sorted_idx]
        sorted_probs  = [proba[i] * 100 for i in sorted_idx]
        bar_colors    = [OUTCOME_COLOR.get(l, "#a855f7") for l in sorted_labels]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#1a1030")

        bars = ax.barh(sorted_labels, sorted_probs, color=bar_colors,
                       edgecolor="none", height=0.65)

        # Highlight predicted bar
        for bar, lbl in zip(bars, sorted_labels):
            if lbl == pred_label:
                bar.set_edgecolor("#ffffff")
                bar.set_linewidth(1.5)

        # Value labels
        for bar, pct in zip(bars, sorted_probs):
            ax.text(pct + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", ha="left",
                    color="#c4b5fd", fontsize=9)

        ax.set_xlabel("Probability (%)", color="#a78bfa", fontsize=10)
        ax.tick_params(colors="#c4b5fd", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.xaxis.label.set_color("#a78bfa")
        ax.set_xlim(0, max(sorted_probs) * 1.25)
        ax.set_title("All outcome probabilities", color="#e8b4ff",
                     fontsize=11, pad=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Insight blurbs ──────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("💡  Profile insights")

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.metric("Swipe selectivity",
                  f"{swipe_right_ratio*100:.0f}%",
                  delta="selective" if swipe_right_ratio < 0.4 else "open")
    with ic2:
        st.metric("Engagement score",
                  f"{int((message_sent_count/100 + emoji_usage_rate)/2*100)}%")
    with ic3:
        st.metric("Profile completeness",
                  f"{int((profile_pics_count/6 + min(bio_length,500)/500)/2*100)}%")
    with ic4:
        st.metric("Activity level",
                  f"{int(app_usage_time_min/300*100)}%")

    # ── Try again nudge ─────────────────────────────────────────────────
    st.info("🔄 Adjust the sliders above and click **Predict** again to see how profile changes affect the outcome.")
