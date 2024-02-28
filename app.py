import streamlit as st
import numpy as np
from prediction import predict



# st.image('images/titanic.png', caption='Sunrise by the mountains')

st.write("Hello World")

# pclass = st.selectbox('What was your passenger class?', [1,2,3])
# sex = st.selectbox('What is your sex?', [1, 0])
# sibsp = st.selectbox('Number of siblings aboard', [0,1,2,3])
# parch = st.selectbox('Number of parents aboard', [0,1,2])

X = np.array([[150.000000, 2727.000000, 24.965357, 2.000000, 0.084458, 18.914684, 1607.000000, 29.000000, 5.000000]])

if st.button('Predict!'):
    result = predict(X)
    st.text(result[0])