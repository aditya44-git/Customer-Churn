import streamlit as st
import pandas as pd
import joblib


model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
num_cols = joblib.load("num_cols.pkl")


def encode_inputs(gender, contract, tenure, monthly_charges):
    gender_map = {"male": 1, "female": 0}
    contract_map = {
        "month-to-month": 0,
        "one year": 1,
        "two year": 2
    }

    return {
        "gender": gender_map[gender],
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": tenure,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 2,
        "OnlineSecurity": 0,
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 1,
        "StreamingMovies": 1,
        "Contract": contract_map[contract],
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": tenure * monthly_charges
    }


def explain_prediction(contract, tenure, monthly_charges, prob):
    reasons = []

    if contract == "month-to-month":
        reasons.append("Short-term contract increases churn risk")
    else:
        reasons.append("Long-term contract reduces churn risk")

    if tenure < 12:
        reasons.append("Low tenure (new customer)")
    else:
        reasons.append("High tenure (loyal customer)")

    if monthly_charges > 80:
        reasons.append("High monthly charges may cause dissatisfaction")
    else:
        reasons.append("Affordable monthly charges support retention")

    summary = (
        "Overall churn risk is HIGH"
        if prob > 0.5
        else "Customer retention factors are STRONG"
    )

    return reasons, summary

# ---------------- STREAMLIT PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1 {
    text-align: center;
    color: #2c3e50;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown("<h1>📊 Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write(
    "Predict whether a telecom customer is **likely to churn or stay**, "
    "using a machine learning model with clear business explanations."
)

# ---------------- INPUT CARD ---------------- #

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔍 Enter Customer Details")

gender = st.selectbox("👤 Gender", ["male", "female"])
contract = st.selectbox(
    "📄 Contract Type",
    ["month-to-month", "one year", "two year"]
)
tenure = st.slider("📆 Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("💰 Monthly Charges", 20, 150, 70)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #

if st.button("🔮 Predict Churn", use_container_width=True):
    input_data = encode_inputs(gender, contract, tenure, monthly_charges)
    input_df = pd.DataFrame([input_data])

    input_df[num_cols] = imputer.transform(input_df[num_cols])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error("🚨 High Risk: Customer is likely to CHURN")
    else:
        st.success("✅ Low Risk: Customer is likely to STAY")

    st.write(f"**Churn Probability:** {probability*100:.2f}%")
    st.progress(int(probability * 100))

    reasons, summary = explain_prediction(
        contract, tenure, monthly_charges, probability
    )

    st.subheader("🧠 Reason for Prediction")
    for r in reasons:
        st.write("•", r)

    st.info(summary)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class="footer">
Built with ❤️ using Machine Learning & Streamlit<br>
Telco Customer Churn Prediction Project
</div>
""", unsafe_allow_html=True)
