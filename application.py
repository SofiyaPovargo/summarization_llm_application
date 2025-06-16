import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Заголовок приложения
st.title("📝 Суммаризация текста")
st.write("Используется модель `csebuetnlp/mT5_multilingual_XLSum` для краткого пересказа текста.")

# Загрузка модели (кешируется, чтобы не загружать при каждом обновлении)
@st.cache_resource
def load_model():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

# Функция для суммаризации (аналогичная вашей)
def summarize_text(text, max_length=300):
    inputs = tokenizer(
        text,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=50,
        num_beams=8,
        repetition_penalty=2.0,
        length_penalty=2.0,
        early_stopping=False,
        temperature=0.7
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Элементы интерфейса
text_input = st.text_area("Введите текст для суммаризации:", height=200)
max_length = st.slider("Максимальная длина вывода:", 50, 300, 150)

if st.button("Суммаризировать"):
    if text_input.strip() == "":
        st.warning("Введите текст!")
    else:
        with st.spinner("Обработка..."):
            summary = summarize_text(text_input, max_length)
        st.subheader("Результат:")
        st.write(summary)

# Инструкция
st.markdown("---")
st.markdown("### Примеры текстов для теста:")
st.code("""Москва — столица России, крупнейший город страны. Здесь находятся Кремль, Красная площадь и другие достопримечательности.""")
st.code("""The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is one of the most famous landmarks in the world.""")