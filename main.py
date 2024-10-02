import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data and cache it for efficiency
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Sidebar for model selection
st.sidebar.title("Choose Model and Input Features")
model_type = st.sidebar.selectbox("Choose Classifier", ["Random Forest", "Support Vector Machine"])

# Sliders for input features
sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Model selection
if model_type == "Random Forest":
    model = RandomForestClassifier()
else:
    model = SVC()

# Train the model
model.fit(df.iloc[:, :-1], df['species'])

# Predict based on user input
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Display the prediction
st.write("## Prediction")
st.write(f"The predicted species is: **{predicted_species}**")

# Feature importance for Random Forest model
if model_type == "Random Forest":
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(df.columns[:-1], feature_importances, color="skyblue")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# Data distribution
st.write("## Data Distribution")
fig, ax = plt.subplots()
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
st.pyplot(fig)

# Model performance metrics
st.write("## Model Performance")
y_pred = model.predict(df.iloc[:, :-1])
accuracy = accuracy_score(df['species'], y_pred)
st.write(f"Accuracy of the {model_type}: **{accuracy:.2f}**")

st.write("## User Feedback")
feedback = st.radio("Is the prediction correct?", ('Yes', 'No'))
if feedback == 'Yes':
    st.write("Thank you for your feedback!")
else:
    st.write("We'll consider this for future improvements.")
