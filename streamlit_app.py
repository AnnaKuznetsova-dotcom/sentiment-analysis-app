import streamlit as st 
import joblib 
import numpy as np 
import pandas as pd

model = None
vectorizer = None

@st.cache_resource(show_spinner=False)
def load_model_data():
    model = open('best_model.pkl', 'rb') # Убрали ./
    model = joblib.load(model)
    vectorizer = open('vectorizer.pkl', 'rb') # Убрали ./
    vectorizer = joblib.load(vectorizer)
    return model, vectorizer

markup = {0: 'Негативная', 1: 'Нейтральная', 2: 'Позитивная'}

st.title("Анализ тональности текста")
text_imput = st.text_input('Введите фразу для семантического анализа', '')

if len(text_imput) > 0:
    st.write('### Введенный текст:')
    st.write(text_imput)
    
    # Загрузка моделей происходит один раз благодаря декоратору выше
    model, vectorizer = load_model_data()
    
    vectorized = vectorizer.transform(np.array([text_imput]))
    prediction = model.predict(vectorized)[0]
    
    st.write('### Предсказанная семантическая окраска текста:')
    st.success(markup[prediction]) # Используем красивую плашку success