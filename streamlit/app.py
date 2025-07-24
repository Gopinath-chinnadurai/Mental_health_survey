import streamlit as st
import torch
import numpy as np
import joblib
import os
import torch.nn as nn

class DepressionMLP(nn.Module):
    def __init__(self, input_size):
        super(DepressionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = DepressionMLP(input_size=11)
model.load_state_dict(torch.load(r"C:\AIML Projects\Mental_health_survey\model\depression_model.pth", map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load(r"C:\AIML Projects\Mental_health_survey\scalers\minmax_scaler.pkl")
encoder_dir = r"C:\AIML Projects\Mental_health_survey\encoders"
encoders = {}
for filename in os.listdir(encoder_dir):
    if filename.endswith("_encoder.pkl"):
        key = filename.replace("_encoder.pkl", "")
        encoders[key] = joblib.load(os.path.join(encoder_dir, filename))

def clean_col_name(col):
    return col.strip().replace(" ", "_").replace("?", "").replace("/", "_")

st.set_page_config(page_title="Mental Health Depression Predictor", layout="centered")
st.title(" Mental Health Depression Predictor")
st.markdown("Fill the form and click **Predict** to assess depression risk.")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
status = st.selectbox("Are you a:", ["Working Professional", "Student"])
work_pressure = st.selectbox("Work Pressure", ["1.0", "2.0", "3.0", "4.0", "5.0"])
job_satisfaction = st.selectbox("Job Satisfaction", ["1.0", "2.0", "3.0", "4.0", "5.0"])
sleep_duration = st.selectbox("Sleep Duration", [
    "Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"
])
diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_hours = st.number_input("Work/Study Hours per Day", min_value=0.0, max_value=24.0, value=6.0)
financial_stress = st.selectbox("Financial Stress (0-5)", ["0", "1", "2", "3", "4", "5"])
family_history = st.radio("Family History of Mental Illness?", ["Yes", "No"])

if st.button(" Predict Depression"):
    try:
        input_data = {
            "Gender": gender,
            "Age": age,
            "Working Professional or Student": status,
            "Work Pressure": work_pressure,
            "Job Satisfaction": job_satisfaction,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": diet,
            "Have you ever had suicidal thoughts ?": suicidal_thoughts,
            "Work/Study Hours": work_hours,
            "Financial Stress": financial_stress,
            "Family History of Mental Illness": family_history
        }

        final_input = []
        for key, value in input_data.items():
            cleaned_key = clean_col_name(key)
            if cleaned_key in encoders:
                encoder = encoders[cleaned_key]
                encoded = encoder.transform([str(value)])[0]
                final_input.append(encoded)
            else:
                final_input.append(float(value)) 

        final_input = np.array([final_input], dtype=np.float32)

        final_input[:, [1, 8, 9]] = scaler.transform(final_input[:, [1, 8, 9]])

        tensor_input = torch.tensor(final_input)

        with torch.no_grad():
            output = model(tensor_input)
            prediction = "Yes" if output.item() >= 0.5 else "No"
            confidence = round(output.item() * 100, 2)

        st.success(f" Facing depression ?  : {prediction}")
        st.info(f" Confidence: {confidence}%")
        st.caption("This prediction is based on your responses. If you're facing any emotional distress, consider seeking help from a mental health professional.")


    except Exception as e:
        st.error(f" Error during prediction: {e}")
