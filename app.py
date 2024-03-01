import streamlit as st
import numpy as np
import pandas as pd
from prediction import predict

st.title("Machine Learning Analysis")

raw_data = pd.read_csv('./streamlit_data/tables/raw_data.csv')
st.table(raw_data)

numeric_data = pd.read_csv('./streamlit_data/tables/numeric_data.csv')
st.table(numeric_data)

st.image('./streamlit_data/img/unshaped_data.png', caption='Data before reshaping')
st.image('./streamlit_data/img/reshaped_data.png', caption='Data after reshaping')

algorithms_metrics = pd.read_csv('./streamlit_data/tables/algoritms_result.csv')
st.table(algorithms_metrics)

transactionid = st.number_input('Transaction ID')
merchantid = st.number_input('Merchant ID')
transaction_amount = st.number_input('Transaction Amount')
category = st.selectbox('Category', [0, 1, 2, 3, 4])
anomaly_score = st.number_input('Anomaly Score')
amount = st.number_input('Amount')
customerid = st.number_input('Customer ID')
age = st.number_input('Age')
hour = st.number_input('Hour')


st.write("Es tiempo de probar el rendimiento de nuestro Algortimo, atraves de unas pruebas con casos reales de nuestro dataset")
st.write("Tenemos eL caso 828 donde el resultado deberia ser 0")
st.write("Ejemplo numero 1: 829, 2524, 50.962744, 2, 0.838278, 98.143387, 1565, 33, 12")
st.write("Tambien tenemos el caso 1420 el cual nos deberia dar un resultado 1")
st.write("Ejemplo numero 2: 150, 2727, 24.965357, 2, 0.084458, 18.914684, 1607, 29, 5")


X = np.array([[transactionid, merchantid, transaction_amount, category, anomaly_score, amount, customerid, age, hour]])

if st.button('Obtener Prediccion'):
    result = predict(X)
    st.text(f'El resultado de nuestra prediccion es:{result[0]}')