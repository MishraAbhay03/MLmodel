import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st


@st.cache_data
def load_data():
    data = pd.read_csv('laptop_price.csv')  
    return data

@st.cache_resource
def train_model(data):
    X = data[['Company', 'Product', 'RAM (GB)', 'Memory', 'CPU_Company', 'OpSys']]
    y = data['Price(Rupee)']
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns  

data = load_data()
model, scaler, feature_columns = train_model(data)

st.title('Laptop Price Prediction')
st.write('Enter the features of the laptop to predict its price.')


company = st.selectbox('Company', data['Company'].unique())
product = st.selectbox('Product', data['Product'].unique())
ram = st.number_input('RAM (GB)', min_value=0, max_value=64, value=8)  # Adjust max_value as needed
memory = st.text_input('Memory (e.g., SSD 512GB, HDD 1TB)')
cpu_company = st.selectbox('CPU Company', data['CPU_Company'].unique())
os = st.selectbox('Operating System', data['OpSys'].unique())

input_data = pd.DataFrame({
    'Company': [company],
    'Product': [product],
    'RAM (GB)': [ram],
    'Memory': [memory],
    'CPU_Company': [cpu_company],
    'OpSys': [os]
})


input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Predicted Price: ₹{prediction[0]:,.2f}')  

