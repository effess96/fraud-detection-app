import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest


# =========================
# 1. DATA + MODEL TRAINING
# =========================

@st.cache_resource
def train_fraud_models(n_transactions: int = 50000):
    """
    Generate synthetic 'Jordan-style' card/payment transactions,
    train a RandomForest fraud classifier + IsolationForest anomaly model,
    and return (feature_cols, clf, iso).
    """
    np.random.seed(42)

    n = n_transactions

    # ---------- Core transaction features ----------
    amount = np.random.exponential(scale=150, size=n)  # many small, few large

    is_new_beneficiary = np.random.binomial(1, 0.18, n)    # ~18% to new payee
    is_international   = np.random.binomial(1, 0.10, n)    # ~10% foreign
    night_time         = np.random.binomial(1, 0.25, n)    # 25% at night
    failed_logins_before = np.random.poisson(lam=0.4, size=n)

    merchant_types = [
        "groceries", "electronics", "online_gaming",
        "atm_withdrawal", "travel", "bill_payment", "restaurants"
    ]
    merchant_type = np.random.choice(
        merchant_types, size=n,
        p=[0.25, 0.14, 0.08, 0.20, 0.08, 0.15, 0.10]
    )

    channels = ["POS", "ECOM", "ATM", "MOBILE"]
    channel = np.random.choice(
        channels, size=n,
        p=[0.45, 0.25, 0.15, 0.15]
    )

    countries = ["JO", "TR", "US", "GB", "AE", "DE", "CN", "EG"]
    country = np.random.choice(
        countries, size=n,
        p=[0.70, 0.06, 0.05, 0.03, 0.05, 0.03, 0.04, 0.04]
    )

    country_risk = (country != "JO").astype(int)  # 1 if non-Jordan

    # ---------- Simulate SIM-swap / mobile risk ----------
    sim_swap_recent = np.random.binomial(1, 0.03, n)   # 3% suspicious SIM event
    otp_misuse_flag = np.random.binomial(1, 0.05, n)   # 5% weird OTP behaviour

    # ---------- Fraud score logic (fake "real world" rules) ----------
    fraud_score = np.zeros(n, dtype=int)

    # high amount = risky
    fraud_score += (amount > 700).astype(int)
    fraud_score += (amount > 1500).astype(int)   # extra bump for very high

    # CNP / ECOM risk
    fraud_score += (channel == "ECOM").astype(int)

    # risky countries and international use
    fraud_score += is_international
    fraud_score += country_risk

    # new beneficiary is risky
    fraud_score += is_new_beneficiary

    # night time risk
    fraud_score += night_time

    # certain merchants more abused
    fraud_score += (merchant_type == "online_gaming").astype(int)
    fraud_score += (merchant_type == "travel").astype(int)

    # behavioural risk
    fraud_score += (failed_logins_before >= 3).astype(int)

    # SIM swap / OTP misuse
    fraud_score += sim_swap_recent
    fraud_score += otp_misuse_flag

    # Convert score → fraud label
    fraud_label = (fraud_score >= 4).astype(int)

    data = pd.DataFrame({
        "amount": amount,
        "is_new_beneficiary": is_new_beneficiary,
        "is_international": is_international,
        "night_time": night_time,
        "failed_logins_before": failed_logins_before,
        "merchant_type": merchant_type,
        "channel": channel,
        "country": country,
        "country_risk": country_risk,
        "sim_swap_recent": sim_swap_recent,
        "otp_misuse_flag": otp_misuse_flag,
        "fraud_score": fraud_score,
        "fraud": fraud_label
    })

    # --- Encode categorical features ---
    data_encoded = pd.get_dummies(
        data.drop(columns=["fraud_score"]),  # drop internal score
        columns=["merchant_type", "channel", "country"],
        drop_first=True
    )

    # separate features/label
    feature_cols = [c for c in data_encoded.columns if c != "fraud"]
    X = data_encoded[feature_cols]
    y = data_encoded["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --- Train main classifier ---
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # --- Train anomaly model on non-fraud data ---
    normal_X_train = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(normal_X_train)

    return feature_cols, clf, iso


def make_transaction_row(
    amount,
    is_new_beneficiary,
    is_international,
    night_time,
    failed_logins_before,
    merchant_type,
    channel,
    country,
    sim_swap_recent,
    otp_misuse_flag,
    feature_cols
):
    base = pd.DataFrame([{
        "amount": amount,
        "is_new_beneficiary": is_new_beneficiary,
        "is_international": is_international,
        "night_time": night_time,
        "failed_logins_before": failed_logins_before,
        "merchant_type": merchant_type,
        "channel": channel,
        "country": country,
        "country_risk": 1 if country != "JO" else 0,
        "sim_swap_recent": sim_swap_recent,
        "otp_misuse_flag": otp_misuse_flag,
        "fraud": 0  # dummy
    }])

    enc = pd.get_dummies(
        base,
        columns=["merchant_type", "channel", "country"],
        drop_first=True
    )

    # drop label if present
    if "fraud" in enc.columns:
        enc = enc.drop(columns=["fraud"])

    # ensure all training columns exist
    for col in feature_cols:
        if col not in enc.columns:
            enc[col] = 0

    # order columns
    enc = enc[feature_cols]

    return enc


def score_transaction(
    amount,
    is_new_beneficiary,
    is_international,
    night_time,
    failed_logins_before,
    merchant_type,
    channel,
    country,
    sim_swap_recent,
    otp_misuse_flag,
    feature_cols,
    clf,
    iso
):
    row_enc = make_transaction_row(
        amount=amount,
        is_new_beneficiary=is_new_beneficiary,
        is_international=is_international,
        night_time=night_time,
        failed_logins_before=failed_logins_before,
        merchant_type=merchant_type,
        channel=channel,
        country=country,
        sim_swap_recent=sim_swap_recent,
        otp_misuse_flag=otp_misuse_flag,
        feature_cols=feature_cols
    )

    # classifier
    clf_prob = float(clf.predict_proba(row_enc)[0][1])

    # anomaly model
    iso_pred = iso.predict(row_enc)[0]   # -1 = anomaly, 1 = normal
    anomaly_flag = 1 if iso_pred == -1 else 0
    anomaly_score = float(anomaly_flag)

    # combine
    final_score = 0.75 * clf_prob + 0.25 * anomaly_score

    # risk level
    if final_score >= 0.8:
        risk_level = "HIGH"
    elif final_score >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # explanation
    reasons = []

    if amount > 1500:
        reasons.append("Very high transaction amount")
    elif amount > 700:
        reasons.append("High transaction amount")

    if is_new_beneficiary:
        reasons.append("New beneficiary")

    if is_international:
        reasons.append("International transaction")

    if country != "JO":
        reasons.append(f"Transaction in foreign country: {country}")

    if channel == "ECOM":
        reasons.append("Card-not-present / e-commerce channel")

    if merchant_type in ["online_gaming", "travel"]:
        reasons.append(f"High-risk merchant type: {merchant_type}")

    if night_time:
        reasons.append("Transaction at night")

    if failed_logins_before >= 3:
        reasons.append("Multiple failed logins before transaction")

    if sim_swap_recent:
        reasons.append("Recent SIM-swap event on customer line")

    if otp_misuse_flag:
        reasons.append("Suspicious OTP behaviour")

    if anomaly_flag:
        reasons.append("Anomaly model flagged this transaction as unusual")

    if not reasons:
        reasons.append("No strong individual risk factors detected; risk from learned patterns")

    return {
        "fraud_probability": round(clf_prob, 3),
        "anomaly_score": anomaly_score,
        "final_score": round(final_score, 3),
        "risk_level": risk_level,
        "reasons": reasons
    }


# =========================
# 2. STREAMLIT UI
# =========================

st.set_page_config(page_title="AI Fraud Detection Demo", page_icon="💳", layout="centered")

st.title("💳 AI Fraud Detection Demo")
st.write(
    "This is a **simulation-based card fraud detector** inspired by Jordan's financial context.\n"
    "It uses synthetic data, a Random Forest classifier, and an Isolation Forest anomaly model."
)

with st.spinner("Training fraud detection models (simulated data)..."):
    feature_cols, clf, iso = train_fraud_models()
st.success("Models ready! You can now score transactions.")

st.header("🧪 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction amount (JOD)", min_value=0.0, max_value=10000.0, value=150.0, step=10.0)
    failed_logins_before = st.slider("Failed logins before transaction", min_value=0, max_value=10, value=0)
    night_time = st.checkbox("Night time transaction (e.g. 23:00–05:00)", value=False)
    is_new_beneficiary = st.checkbox("New beneficiary", value=False)

with col2:
    is_international = st.checkbox("International transaction", value=False)

    merchant_type = st.selectbox(
        "Merchant type",
        ["groceries", "electronics", "online_gaming",
         "atm_withdrawal", "travel", "bill_payment", "restaurants"]
    )

    channel = st.selectbox(
        "Channel",
        ["POS", "ECOM", "ATM", "MOBILE"]
    )

    country = st.selectbox(
        "Country of transaction",
        ["JO", "TR", "US", "GB", "AE", "DE", "CN", "EG"]
    )

    sim_swap_recent = st.checkbox("Recent SIM-swap on customer line", value=False)
    otp_misuse_flag = st.checkbox("Suspicious OTP behaviour", value=False)

st.markdown("---")

if st.button("🔍 Score Transaction"):
    result = score_transaction(
        amount=amount,
        is_new_beneficiary=int(is_new_beneficiary),
        is_international=int(is_international),
        night_time=int(night_time),
        failed_logins_before=int(failed_logins_before),
        merchant_type=merchant_type,
        channel=channel,
        country=country,
        sim_swap_recent=int(sim_swap_recent),
        otp_misuse_flag=int(otp_misuse_flag),
        feature_cols=feature_cols,
        clf=clf,
        iso=iso
    )

    st.subheader("📊 Fraud Risk Result")

    colA, colB, colC = st.columns(3)
    colA.metric("Risk level", result["risk_level"])
    colB.metric("Model fraud probability", f"{result['fraud_probability']:.3f}")
    colC.metric("Final combined score", f"{result['final_score']:.3f}")

    st.write("### 🧾 Reasons / Signals")
    for r in result["reasons"]:
        st.markdown(f"- {r}")

    st.info("Note: This is a **simulation / learning demo**, not a production model.")
else:
    st.info("Fill the fields above and click **Score Transaction** to see the risk assessment.")
