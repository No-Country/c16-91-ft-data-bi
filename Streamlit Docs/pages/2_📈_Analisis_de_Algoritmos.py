import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Análisis de Algoritmos",
    page_icon=":chart_with_upwards_trend:",
)

st.title('Análisis de Algoritmos')
st.markdown(
    '''
    Para nuestro análisis de algoritmos consideramos usar la siguiente lista:  
    - Logistic Regression  
    - Naive Bayes  
    - K Nearest Neighborg  
    - Support Vector Machine  
    - Decision Trees  
    
    Estos son algoritmos los cuales son usados en **Supervised Learning**, especificamente en clasificación, el cual es nuestro caso.
    Consideramos tambien que las mejores metricas para evaluar nuestros algoritmos son la exactitud (**Accuracy**) del algoritmo y el **F1 Score**, 
    el cual es la relación que se tiene entre la precision y el recall de cada algoritmo, si el valor es mas cercano a 1, el algoritmo es mejor.
    '''
)

algorithms_metrics = pd.read_csv('./streamlit_data/tables/algoritms_result.csv')
st.table(algorithms_metrics)

st.markdown(
    '''
    Segun los resultados de nuestra tabla, podemos decir que el algoritmo que tiene las mejores metricas es el de Support Vector Machine (SVM).  
    Si quieres aprender más sobre el algoritmo de SVM, puedes ver este video:  
    '''
)

st.video('https://www.youtube.com/watch?v=_YPScrckx28')

st.write('Créditos a Visually Explained, autor del video.')