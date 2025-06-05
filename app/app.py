import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DepressionPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DepressionPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

st.title("Depression Prediction App")

st.markdown("""  
Upload your CSV file with required features to get predictions.
""")

st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload preprocessed CSV", type=["csv"])

if st.sidebar.button("Download Sample Input CSV"):
    sample_df = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Working Professional or Student': ['Student', 'Professional'],
        'Profession': ['Student', 'Engineer'],
        'Dietary Habits': ['Vegetarian', 'Non-Vegetarian'],
        'Degree': ['Bachelor', 'Master'],
        'Sleep Duration': [7, 6],
        'Have you ever had suicidal thoughts ?': ['No', 'Yes'],
        'Family History of Mental Illness': ['No', 'Yes'],
        'Age': [25, 30],
        'Academic Pressure': [3, 2],
        'Work Pressure': [2, 3],
        'CGPA': [7.8, 8.4],
        'Study Satisfaction': [4, 3],
        'Job Satisfaction': [3, 4],
        'Work/Study Hours': [5, 6],
        'Financial Stress': [2, 3]
    })
    csv_sample = sample_df.to_csv(index=False)
    st.sidebar.download_button("Download Sample CSV", csv_sample, "sample_input.csv", "text/csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head(5))

    st.markdown(f"**Data shape:** {data.shape[0]} rows and {data.shape[1]} columns")

    categorical_columns = [
        'Gender', 'Working Professional or Student', 'Profession', 'Dietary Habits',
        'Degree', 'Sleep Duration', 'Have you ever had suicidal thoughts ?',
        'Family History of Mental Illness'
    ]
    numerical_columns = [
        'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
        'Financial Stress'
    ]

    all_features = categorical_columns + numerical_columns

    missing_cols = [col for col in all_features if col not in data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    X = data[all_features].copy()

    scaler = StandardScaler()
    scaler.mean_ = np.array([28.90, 2.45, 3.01, 7.35, 1.45, 2.31, 6.9, 2.95])  
    scaler.scale_ = np.array([9.89, 0.5, 1.6, 1.01, 0.65, 0.75, 2.7, 1.1])    

   
    with st.spinner("Preprocessing data and making predictions..."):
        X[numerical_columns] = scaler.transform(X[numerical_columns])

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        input_dim = X_tensor.shape[1]
        model = DepressionPredictor(input_dim)
        model.load_state_dict(torch.load(r'C:\AIML Projects\Mental_health_survey\model\depression_model.pth', map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            predictions = model(X_tensor).squeeze().numpy()

    
    st.subheader("Prediction Results")
    result_df = data.copy()
    result_df['Depression_Probability'] = predictions
    result_df['Predicted_Depression'] = (predictions >= 0.5).astype(int)

    st.dataframe(result_df[['Depression_Probability', 'Predicted_Depression']].head(10))

  
    csv = result_df.to_csv(index=False)
    st.download_button("Download Full Predictions", csv, "depression_predictions.csv", "text/csv")

else:
    st.info("Please upload a preprocessed CSV file to start predictions.")
