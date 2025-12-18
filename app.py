import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Iris Flower Prediction", layout="centered")

st.title("🌸 Iris Flower Prediction App")
st.write("Predict the **Iris flower species** using an SVM model")

# Load the trained model
@st.cache_resource
def load_model():
    with open("svm_iris_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("✅ Model loaded successfully")

# User input
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    species = {
        0: "🌼 Iris Setosa",
        1: "🌺 Iris Versicolor",
        2: "🌸 Iris Virginica"
    }

    st.subheader("Prediction Result")
    st.success(f"Predicted Species: **{species[prediction]}**")
