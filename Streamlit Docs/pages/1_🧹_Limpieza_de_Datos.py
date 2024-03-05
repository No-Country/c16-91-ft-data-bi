import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Limpieza de Datos",
    page_icon=":broom:",
)

st.title("Limpieza de Datos")

st.markdown(
    '''         
    A partir de un análisis de los datos (EDA) se genera una tabla con todos los features (columnas) más significativos, 
    asi como también nuestros targets (Columna Fraud Indicator) los cuales deberiamos poder predecir. 
    La tabla que se muestra a continuación, contiene 7 casos aleatorios de nuestro dataset completo.   
    Como podemos observar algunas columnas contienen distintos tipos de datos. Por ejemplo, tenemos una columna de Timestamp la cual tiene tipo de dato fecha. 
    También tenemos una columna con categorias que son de tipo string. Sin embargo necesitamos que todos nuestros sean de **tipo numérico**, para que de esta forma
    nuestros algoritmos puedan aprender de una forma más eficiente.

    '''
)

raw_data = pd.read_csv('./streamlit_data/tables/raw_data.csv')
st.table(raw_data)

st.markdown(
    '''
    Luego de aplicar distintas funciones y técnicas mostramos otros 7 casos aleatorios, en donde observamos que solo hay datos numéricos.
    Ahora, debemos tener en cuenta que los rangos entre nuestros datos varian mucho. 
    Por ejemplo, si observamos las columnas MerchantID y Anomaly Score estas difieren mucho su rango. Para esto se debe aplicar un scaler que nos permita **normalizar** los datos.
    '''
)

numeric_data = pd.read_csv('./streamlit_data/tables/numeric_data.csv')
st.table(numeric_data)

st.markdown(
    '''
    Otro detalle que podemos observar, es que la cantidad de casos fraudulentos vs la cantidad de casos no fraudulentes esta muy desbalanceada. 
    Tenemos 955 casos no fraudulentos y solo 45 fraudulentos. Es necesario hacer un **resize** de estos datos, esto ayudará a nuestros algoritmos a que 
    aprendan a distinguir entre los casos 0 y 1.
    '''
)

st.image('./streamlit_data/img/unshaped_data.png', caption='Data antes de hacer un resize')
st.image('./streamlit_data/img/reshaped_data.png', caption='Data despúes de hacer un resize')

st.markdown(
    '''
    Una vez que nuestros datos son de tipo numérico, estan normalizados y estan balanceados, se puede proceder a hacer un mejor análisis por distintos algoritmos :wink:.

    '''
)