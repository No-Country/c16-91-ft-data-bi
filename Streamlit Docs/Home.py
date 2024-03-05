import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon=":house:",
)

st.image('./streamlit_data/img/company_logo.png')

st.title("Bienvenid@ a nuestra página de análisis de Machine Learning👋")

st.markdown(
    '''
    Esta es una web app construida con Streamlit, en podrás encontrar toda la información sobre el análisis de Machine Learning que realizamos en nuestro proyecto.  

    Nuestro análisis se divide en 3 secciones principales:  

    - Limpieza de Datos
    - Análisis de Algoritmos
    - Predicción

    Por favor selecciona una de las secciones de la barra lateral, para ver los pasos que se siguieron.  
    
    Para tener una vista completa de nuestro trabajo, puedes ver el documento completo en [Google Colab](https://colab.research.google.com/drive/12pULlDpbJ9coJzNxDaUDk0KMYXz7wdKC?usp=sharing).  
    
    O también puedes volver a nuestra [página web](https://fintech-solvers.netlify.app/).
    '''
)