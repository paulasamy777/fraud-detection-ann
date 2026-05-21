import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =========================
# Load model and scaler
# =========================
model = load_model('fraud_model.h5')
scaler = joblib.load('scaler.pkl')

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

st.title("💳 AI Fraud Detection System")
st.write("Enter all transaction details (30 features model).")

# =========================
# Inputs (30 FEATURES)
# =========================
time = st.number_input("Time (seconds elapsed)", value=0.0, min_value=0.0)

st.markdown("#### PCA Components (V1 – V28)")
cols = st.columns(4)
v_vals = []
for i in range(1, 29):
    col = cols[(i - 1) % 4]
    v_vals.append(col.number_input(f"V{i}", value=0.0, key=f"v{i}"))

amount = st.number_input("Amount ($)", min_value=0.0, value=0.0)

# =========================
# Threshold slider
# =========================
st.markdown("---")
threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.3,
                      help="Lower = catch more fraud. Higher = fewer false alarms.")
st.caption("💡 Default 0.3 — optimized for fraud recall.")

# =========================
# Predict
# =========================
if st.button("🔍 Predict Fraud", use_container_width=True):

    # Scale Amount exactly like training (Time was NOT scaled in training)
    amount_scaled = scaler.transform([[amount]])[0][0]

    input_data = np.array([[
        time,          # ← raw, not scaled (matches your notebook)
        *v_vals,
        amount_scaled  # ← scaled with the real scaler
    ]], dtype=float)

    prediction = model.predict(input_data, verbose=0)
    fraud_probability = float(prediction[0][0])

    # ── Results ──
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("Fraud Probability", f"{fraud_probability:.2%}")
    col2.metric("Threshold Used", f"{threshold:.2f}")

    st.progress(fraud_probability, text="Risk Level")

    if fraud_probability > threshold:
        st.error("⚠️ Fraudulent Transaction Detected!")
        st.warning(f"Confidence: {fraud_probability:.2%} exceeds threshold of {threshold:.0%}")
    else:
        st.success("✅ Legitimate Transaction")
        st.info(f"Fraud score {fraud_probability:.2%} is below threshold of {threshold:.0%}")