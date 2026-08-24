import streamlit as st
import joblib
import numpy as np
import pandas as pd

markup = {0: 'Негативная', 1: 'Нейтральная', 2: 'Позитивная'}

# Функция загружает модель ТОЛЬКО один раз при старте или изменении файлов .pkl
@st.cache_resource(show_spinner=False)
def load_model_data():
    model_file = open('best_model.pkl', 'rb')
    model = joblib.load(model_file)
    
    vectorizer_file = open('vectorizer.pkl', 'rb')
    vectorizer = joblib.load(vectorizer_file)
    
    return model, vectorizer

st.title("Анализ тональности текста")
st.caption("Модель определяет окраску введенной фразы: негативная, нейтральная или позитивная.")

text_imput = st.text_input('Введите фразу для семантического анализа', '', placeholder="Например: Мне очень понравился этот сервис")

if text_imput and len(text_imput.strip()) > 0:
    try:
        st.write('**Введенный текст:**')
        st.write(f'"{text_imput}"')
        
        # Загрузка происходит здесь. Если кэш жив, декоратор вернет данные мгновенно.
        model, vectorizer = load_model_data()
        
        # Преобразование текста в числа
        vectorized = vectorizer.transform(np.array([text_imput]))
        
        # Предсказание
        prediction = model.predict(vectorized)[0]
        
        # Вывод результата
        st.write('**Предсказанная семантическая окраска текста:**')
        st.success(markup[prediction])
        
    except Exception as e:
        st.error(f"Произошла ошибка при обработке данных: {e}")