import streamlit as st
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict
from process import process_eggs
import torch


# fixing bug that appears while using streamlit with torch
torch.classes.__path__ = []


@st.cache_resource
def load_model(config: Dict) -> YOLO:
    return YOLO(config['model_path'])


@st.cache_resource
def load_config() -> Dict:
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    st.title('Анализатор яиц 🥚')
    st.subheader('Загрузите изображение для анализа')

    config = load_config()
    model = load_model(config)

    with st.sidebar:
        st.header('Настройки')
        with st.form('slider_form'):
            config['brightness_threshold'] = st.slider(
                'Порог яркости',
                0, 255, config['brightness_threshold'],
                help='Пороговое значение для классификации на светлые/темные яйца'
            )
            config['conf_threshold'] = st.slider(
                'Порог уверенности',
                0.0, 1.0, config['conf_threshold'],
                help='Минимальная уверенность модели для детекции'
            )
            submit = st.form_submit_button("Применить значения порогов")

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner('Обработка изображения...'):
            annotated_img, white, dark = process_eggs(config, model, image)

        original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image_rgb,
                     caption='Исходное изображение',
                     use_container_width=True)

        with col2:
            st.image(annotated_image_rgb,
                     caption='Результат обработки',
                     use_container_width=True)

        st.success(f"**Результаты анализа:**")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric(label="Светлые яйца", value=white)
        with col_stat2:
            st.metric(label="Темные яйца", value=dark)


if __name__ == '__main__':
    main()
