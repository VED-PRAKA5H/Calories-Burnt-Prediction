import streamlit as st
import pandas as pd
import pickle
import os

# Custom CSS for aesthetics (optional)
page_bg_style = '''
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #050C9C, #3572EF);
        background-size: cover;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.5);
    }
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px #000000;
    }
</style>
'''
st.markdown(page_bg_style, unsafe_allow_html=True)

model_path = f"{os.path.dirname(os.path.realpath(os.path.abspath(__file__)))}" \
             f"/notebook/data/model_pipeline.pkl"

with open(model_path, 'rb') as file:
    model = pickle.load(file)

demo_input = {'Gender': 'female',
              'Age': 45,
              'Height': 154.0,
              'Weight': 52.0,
              'Duration': 26.0,
              'Heart_Rate': 107.0,
              'Body_Temp': 40.6
              }

# Create a Pandas DataFrame
df = pd.DataFrame([demo_input.values()], columns=list(demo_input.keys()))

# Title
st.title("✨ Calories Burn Prediction ✨")

# Separate for this to fit better visually
df.Gender = st.selectbox("Enter Gender", ["male", "female"])

# Side-by-side inputs using columns
col1, col2 = st.columns(2)

with col1:
    df.Age = st.number_input("Enter Age:", value=demo_input['Age'])
    df.Duration = st.number_input("Enter Workout Duration:", value=demo_input['Duration'])
    df.Height = st.number_input("Enter Height (cm):", value=demo_input['Height'])

with col2:
    df.Weight = st.number_input("Enter Weight (kg):", value=demo_input['Weight'])
    df.Heart_Rate = st.number_input("Enter Heart Rate:", value=demo_input['Heart_Rate'])
    df.Body_Temp = st.number_input("Enter Body Temperature (°C):", value=demo_input['Body_Temp'])

# Button to calculate result
if st.button("Show Result"):
    result = model.predict(df)[0]
    st.subheader("🔥 Prediction Result:")
    st.write(f"Based on your inputs, the burned calories are: **{result:.2f} kcal**")
else:
    st.write("_Click the button to see your prediction!_")
