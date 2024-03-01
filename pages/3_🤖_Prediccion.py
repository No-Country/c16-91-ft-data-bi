import streamlit as st
import numpy as np
from prediction import predict

st.set_page_config(
    page_title="Predicción",
    page_icon=":robot_face:",
)

st.title('Predicción')

st.markdown('''
            Es tiempo de probar el rendimiento de nuestro Algoritmo, atraves de unas pruebas con casos reales. Puedes copiar los valores que se 
            muestran a continuación para verificar el rendimiento de nuestro algoritmo. También puedes hacer la prueba con casos de la segunda tabla, en la sección de Limpieza de Datos.

            ### Caso que no es fraudulento (0)  
            Para un caso que no es considerado como fraude tenemos los siguientes valores:     
            -Transaction ID: 829  
            -Merchant ID: 2524  
            -Transaction Amount: 50.962744  
            -Category: 2  
            -Anomaly Score: 0.838278  
            -Amount: 98.143387  
            -Customer ID: 1565  
            -Age: 33  
            -Hour: 12  

            ### Caso que si es fraudulento (1)  

            Para un caso que si es detectado como fraude, tenemos los siguientes valores:    
            Estos son los valores que deberias usar:  
            -Transaction ID: 150  
            -Merchant ID: 2727  
            -Transaction Amount: 24.965357  
            -Category: 2  
            -Anomaly Score: 0.084458  
            -Amount: 18.914684  
            -Customer ID: 1607  
            -Age: 29  
            -Hour: 5  
            ''')

transactionid = st.number_input('Transaction ID', min_value=0)
merchantid = st.number_input('Merchant ID', min_value=0)
transaction_amount = st.number_input('Transaction Amount')
category = st.selectbox('Category', [0, 1, 2, 3, 4])
anomaly_score = st.number_input('Anomaly Score')
amount = st.number_input('Amount')
customerid = st.number_input('Customer ID', min_value=0)
age = st.number_input('Age', min_value=0)
hour = st.number_input('Hour', min_value=0)

X = np.array([[transactionid, merchantid, transaction_amount, category, anomaly_score, amount, customerid, age, hour]])

if st.button('Obtener Prediccion'):
    result = predict(X)
    st.text(f'El resultado de nuestra predicción es: {result[0]}')