import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------------------------------
# STREAMLIT TITLE
# ---------------------------------------
st.title("KNN Regression App")

st.write("This app demonstrates KNN Regression using Scikit-learn.")

# ---------------------------------------
# CREATE DATASET
# ---------------------------------------
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=10,
    random_state=42
)

# convert to dataframe
df = pd.DataFrame(X, columns=["Feature"])
df["Target"] = y

st.subheader("Dataset")
st.dataframe(df.head())

# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# FEATURE SCALING
# ---------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------
# SIDEBAR PARAMETERS
# ---------------------------------------
st.sidebar.header("Model Parameters")

n_neighbors = st.sidebar.slider(
    "Number of Neighbors (K)",
    min_value=1,
    max_value=20,
    value=5
)

p_value = st.sidebar.selectbox(
    "P Value",
    [1, 2]
)

# ---------------------------------------
# MODEL
# ---------------------------------------
model = KNeighborsRegressor(
    n_neighbors=n_neighbors,
    metric='minkowski',
    p=p_value
)

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# ---------------------------------------
# EVALUATION
# ---------------------------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluation Metrics")

st.write("Mean Squared Error:", mse)
st.write("Mean Absolute Error:", mae)
st.write("R2 Score:", r2)

# ---------------------------------------
# USER INPUT PREDICTION
# ---------------------------------------
st.subheader("Make Prediction")

user_input = st.number_input("Enter Feature Value")

if st.button("Predict"):

    input_data = np.array([[user_input]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Value: {prediction[0]:.2f}")

# ---------------------------------------
# SHOW ACTUAL VS PREDICTED
# ---------------------------------------
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.subheader("Actual vs Predicted")
st.dataframe(results.head(10))