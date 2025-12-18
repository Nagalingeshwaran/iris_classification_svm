import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Iris Flower Prediction", layout="centered")

st.title("🌸 Iris Flower Prediction App")

@st.cache_resource
def load_model():
    with open("svm_iris_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    st.success(f"🌼 Predicted Species: **{prediction}**")
