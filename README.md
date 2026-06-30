#  California House Price Prediction using Machine Learning

A machine learning web application that predicts California house prices using a Random Forest Regression model. The project includes data preprocessing, model training, evaluation, visualization, and a Streamlit web interface.

---

##  Features

- Predict house prices based on user input
- Interactive Streamlit web application
- Random Forest Regression model
- Model performance metrics
- Feature importance visualization
- Cross-validation
- Clean and responsive UI

---

##  Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Joblib

---

##  Project Structure

```
house-price-prediction-ml/
│
├── app/
│   └── app.py
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── reports/
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Dataset

- California Housing Dataset
- Source: Scikit-learn (`fetch_california_housing()`)

Features used:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

---

##  Machine Learning Model

Algorithm:
- Random Forest Regressor

Training/Test Split:
- 80% Training
- 20% Testing

Cross Validation:
- 5-Fold Cross Validation

---

##  Model Performance

| Metric | Value |
|---------|------:|
| R² Score | 0.805 |
| MAE | 0.328 |
| RMSE | 0.506 |

---

##  Running the Project

### Clone Repository

```bash
git clone https://github.com/AvishiRoy/house-price-prediction-ml.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python src/train.py
```

### Launch Streamlit App

```bash
python -m streamlit run app/app.py
```

---

##  Application Preview

The Streamlit application allows users to:

- Enter housing details
- Predict estimated house prices
- View model performance
- Explore feature importance
- Understand project workflow

---

##  Developer

**Avishi Roy**

B.Tech Computer Science Engineering  
SRM Institute of Science and Technology

---

##  License

This project is developed for educational purposes.