import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:
    st.title("📌 Project Information")

    st.markdown("""
### 🏠 House Price Prediction

**Model**
- Random Forest Regressor

**Dataset**
- California Housing Dataset (Scikit-Learn)

**Programming Language**
- Python

**Libraries Used**
- Streamlit
- Scikit-Learn
- Pandas
- Matplotlib
- Joblib

**Developer**
- Avishi Roy
""")

# ---------------- LOAD MODEL ---------------- #

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR.parent / "models" / "house_price_model.pkl")

#model = joblib.load("../models/house_price_model.pkl")

# ---------------- TITLE ---------------- #

st.title("🏠 California House Price Prediction")

st.markdown(
"""
Predict estimated house prices using a trained **Random Forest Machine Learning model**.

Enter the property details below and click **Predict House Price**.
"""
)

st.divider()

# ---------------- INPUTS ---------------- #

left, right = st.columns(2)

with left:
    medinc = st.number_input("Median Income", value=5.0)
    houseage = st.number_input("House Age", value=20.0)
    averooms = st.number_input("Average Rooms", value=6.0)
    avebedrms = st.number_input("Average Bedrooms", value=1.0)

with right:
    population = st.number_input("Population", value=1000.0)
    aveoccup = st.number_input("Average Occupancy", value=3.0)
    latitude = st.number_input("Latitude", value=37.0)
    longitude = st.number_input("Longitude", value=-122.0)

st.write("")

# ---------------- PREDICTION ---------------- #

if st.button("🔮 Predict House Price", use_container_width=True):

    data = pd.DataFrame({
        "MedInc":[medinc],
        "HouseAge":[houseage],
        "AveRooms":[averooms],
        "AveBedrms":[avebedrms],
        "Population":[population],
        "AveOccup":[aveoccup],
        "Latitude":[latitude],
        "Longitude":[longitude]
    })

    prediction = model.predict(data)[0]

    st.subheader("Input Summary")

    st.dataframe(data, use_container_width=True)

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="🏡 Estimated House Price",
        value=f"${prediction*100000:,.0f}"
    )

st.divider()

# ---------------- PERFORMANCE ---------------- #

st.header("📈 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("R² Score", "0.805")
c2.metric("MAE", "0.328")
c3.metric("RMSE", "0.506")

st.divider()

# ---------------- MODEL DETAILS ---------------- #

st.header("⚙ Model Details")

info1, info2 = st.columns(2)

with info1:
    st.markdown("""
**Algorithm**
- Random Forest Regressor

**Dataset**
- California Housing Dataset

**Training/Test Split**
- 80% / 20%
""")

with info2:
    st.markdown("""
**Number of Features**
- 8

**Cross Validation**
- 5 Fold

**Model Saved Using**
- Joblib
""")

st.divider()

# ---------------- FEATURE IMPORTANCE ---------------- #

st.header("📊 Feature Importance")

importance = pd.DataFrame({
    "Feature":[
        "Median Income",
        "Average Occupancy",
        "Latitude",
        "Longitude",
        "House Age",
        "Average Rooms",
        "Population",
        "Average Bedrooms"
    ],
    "Importance":[
        0.525,
        0.134,
        0.089,
        0.088,
        0.055,
        0.044,
        0.031,
        0.030
    ]
})

importance = importance.sort_values("Importance")

fig, ax = plt.subplots(figsize=(10,5))

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance Ranking")

st.pyplot(fig)

st.divider()

# ---------------- ABOUT ---------------- #

st.header("ℹ About This Project")

st.markdown("""
This application predicts **California house prices** using a trained
**Random Forest Regression** model.

### Workflow

- Data Collection
- Data Preprocessing
- Train-Test Split
- Random Forest Training
- Model Evaluation
- Cross Validation
- Model Saving using Joblib
- Deployment using Streamlit

---

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- Matplotlib
- Joblib
""")

st.caption("Developed by Avishi Roy • Machine Learning Project")