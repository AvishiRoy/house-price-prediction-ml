import os
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import joblib
import os
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def load_data():
    housing = fetch_california_housing()

    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df["Price"] = housing.target

    X = df.drop("Price", axis=1)
    y = df["Price"]

    return X, y


def train_model(X_train, y_train):
    model=RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance")
    print(importance)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    print("\n========== MODEL PERFORMANCE ==========")
    print(f"R² Score : {r2:.3f}")
    print(f"MAE      : {mae:.3f}")
    print(f"RMSE     : {rmse:.3f}")

    # -----------------------------
    # Actual vs Predicted Plot
    # -----------------------------

    plt.figure(figsize=(8,6))

    plt.scatter(y_test, predictions)

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")

    plt.savefig("reports/actual_vs_predicted.png")

    #plt.show()

    errors = y_test - predictions

    plt.figure(figsize=(8,6))

    plt.scatter(predictions, errors)

    plt.axhline(y=0, color="red")

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")

    plt.title("Residual Plot")

    plt.savefig("reports/residual_plot.png")
    return predictions


def save_model(model):
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/house_price_model.pkl")

    print("\n✅ Model saved successfully!")


def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
    )

    print("\n===== CROSS VALIDATION =====")
    print("Scores:", scores)
    print("Average:", scores.mean())

    save_model(model)


if __name__ == "__main__":
    main()